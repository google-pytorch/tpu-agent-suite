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
"""Unit tests for tpu_slots.py, against a parallel.json in a temp directory."""

from __future__ import annotations

import importlib.util
import io
import json
import multiprocessing
import os
import pathlib
import sys
import tempfile
import unittest
import unittest.mock

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent

try:
    from . import tpu_slots  # noqa: E402
except ImportError:
    _spec = importlib.util.spec_from_file_location(
        "tpu_slots", _SCRIPTS_DIR / "tpu_slots.py")
    tpu_slots = importlib.util.module_from_spec(_spec)
    sys.modules["tpu_slots"] = tpu_slots
    _spec.loader.exec_module(tpu_slots)

IMAGE = "registry.example/vllm@sha256:abc"


def _gke(accelerator="tpu7x",
         topology="2x2x1",
         image=IMAGE,
         chips=4,
         slots=None):
    params = {
        "tpu_accelerator": accelerator,
        "tpu_topology": topology,
        "image": image,
        "tpu_chips": chips
    }
    if slots is not None:
        params["slots"] = slots
    return {"backend": "gke_job", "params": params}


PROFILES = {
    "v6e-a": {
        "backend": "gce_tpu_vm",
        "params": {}
    },
    "v6e-b": {
        "backend": "gce_tpu_vm",
        "params": {}
    },
    "gke": {
        "backend": "gke_job",
        "params": {}
    },
    "here": {
        "backend": "local",
        "params": {}
    },
}


def _claim_in_process(path: str, holder: str, out) -> None:
    """One process's claim, for the concurrency test: a slot name or ''."""
    try:
        out.put(tpu_slots.claim(pathlib.Path(path), holder)["resource"])
    except tpu_slots.Refused:
        out.put("")


class SlotCase(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = pathlib.Path(self.tmp.name) / "parallel.json"

    def state(self):
        return json.loads(self.path.read_text(encoding="utf-8"))

    def open(self, profiles=PROFILES, seen=None, **kw):
        return tpu_slots.open_slots(
            self.path,
            profiles,
            inspect=seen.get if isinstance(seen, dict) else
            (seen or (lambda _: None)),
            **kw)


class TestOpenSlots(SlotCase):

    def test_slots_are_what_each_profile_declares_never_measured(self):
        inspected = []

        def seen(name):
            inspected.append(name)
            # A full Kueue pool: no free chips at all.
            return {"available_chips": 0, "details": {"nodes": ["n1"]}}

        out = self.open(
            {
                "vm": PROFILES["v6e-a"],
                "one": _gke(),
                "two": _gke("tpu-v6e-slice", "2x4", chips=8, slots=2),
                "here": PROFILES["here"],
            },
            seen=seen)
        # A VM is one slot, a gke_job profile one unless it declares more,
        # local none; zero free chips change nothing, and only the VM is
        # inspected (for its hardware, which its profile does not name).
        self.assertEqual(sorted(self.state()["resources"]),
                         ["one-1", "two-1", "two-2", "vm"])
        self.assertEqual((out["parallel"], inspected), (4, ["vm"]))

    def test_slots_are_set_once_and_kept(self):
        self.open({"v6e-a": PROFILES["v6e-a"]})
        tpu_slots.claim(self.path, "sweep:a")
        before = self.state()
        out = self.open({**PROFILES, "gke": _gke(slots=3)})
        self.assertEqual((out["added"], self.state()), ([], before))
        # No TPU profile: ask, and write no file a later claim would trust.
        self.path.unlink()
        with self.assertRaises(tpu_slots.NeedsInput):
            self.open({"here": PROFILES["here"]})
        self.assertFalse(self.path.exists())

    def test_a_hardware_key_is_accelerator_topology_chips_and_image(self):
        no_chips = _gke()
        del no_chips["params"]["tpu_chips"]
        self.assertEqual(tpu_slots.hardware_key(no_chips),
                         f"tpu7x|2x2x1|4|{IMAGE}")
        self.assertEqual(tpu_slots.hardware_key(_gke(chips=4)),
                         tpu_slots.hardware_key(no_chips))
        self.assertNotEqual(tpu_slots.hardware_key(_gke(chips=2)),
                            tpu_slots.hardware_key(no_chips))
        self.assertIsNone(tpu_slots.hardware_key(_gke(image="")))
        vm = {"backend": "gce_tpu_vm"}
        self.assertIsNone(
            tpu_slots.hardware_key(vm, {
                "accelerator": "none",
                "topology": "none"
            }))
        key = tpu_slots.hardware_key(vm, {
            "accelerator": "v6e",
            "topology": "2x2",
            "chips": 4
        })
        self.assertEqual(key, "v6e|2x2|4|-")
        self.assertEqual(tpu_slots.Hardware.parse(key).chips, "4")
        self.assertIsNone(tpu_slots.Hardware.parse("a|b"))


class TestClaim(SlotCase):

    def test_claims_take_free_slots_and_refuse_what_is_held(self):
        self.open()
        first = tpu_slots.claim(self.path, "sweep:a")
        self.assertEqual(tpu_slots.claim(self.path, "sweep:a"), first)
        self.assertNotEqual(
            tpu_slots.claim(self.path, "sweep:b")["resource"],
            first["resource"])
        with self.assertRaises(tpu_slots.Refused):
            tpu_slots.claim(self.path, "sweep:c", resource=first["resource"])
        with self.assertRaises(tpu_slots.NeedsInput):
            tpu_slots.claim(self.path, "sweep:c", resource="nope")
        # gke-1 and v6e-a are held: only v6e-b fits the profile asked for.
        self.assertEqual(
            tpu_slots.claim(self.path, "sweep:c", profile="v6e-b")["resource"],
            "v6e-b")
        with self.assertRaises(tpu_slots.Refused):
            tpu_slots.claim(self.path, "sweep:d")
        with self.assertRaises(tpu_slots.NeedsInput):
            tpu_slots.claim(pathlib.Path(self.tmp.name) / "none.json", "x")

    def test_release_and_set_job_belong_to_the_holder(self):
        self.open()
        name = tpu_slots.claim(self.path, "sweep:a")["resource"]
        tpu_slots.set_job(self.path, name, "sweep:a", "job-1")
        self.assertEqual(self.state()["resources"][name]["job_id"], "job-1")
        with self.assertRaises(tpu_slots.Refused):
            tpu_slots.set_job(self.path, name, "sweep:b", "job-2")
        self.assertFalse(
            tpu_slots.release(self.path, name, "sweep:b")["released"])
        self.assertTrue(
            tpu_slots.release(self.path, name, "sweep:a")["released"])
        self.assertIsNone(self.state()["resources"][name]["job"])

    def test_most_available_takes_the_kind_with_room_for_the_most(self):
        self.open({
            "a": _gke(),
            "big": _gke(slots=2),
            "v6e": _gke("tpu-v6e-slice", "2x4", chips=8, slots=2),
        })
        # tpu7x: 1 + 2 slots -> room for 3; v6e 2. Its profile with the most.
        got = tpu_slots.claim(self.path, "leader", most_available=True)
        self.assertEqual((got["hardware"], got["profile"]),
                         (f"tpu7x|2x2x1|4|{IMAGE}", "big"))
        self.assertEqual(got["free_like_this"], 2)
        follower = tpu_slots.claim(self.path, "f1", hardware=got["hardware"])
        self.assertEqual(follower["hardware"], got["hardware"])
        self.path.unlink()
        self.open({"gke": {"backend": "gke_job", "params": {"slots": 2}}})
        with self.assertRaises(tpu_slots.Refused):
            tpu_slots.claim(self.path, "leader", most_available=True)

    def test_concurrent_claims_never_share_a_slot(self):
        self.open({**PROFILES, "gke": _gke(slots=4)})
        slots = len(self.state()["resources"])
        ctx = multiprocessing.get_context("spawn")
        out = ctx.Queue()
        workers = [
            ctx.Process(target=_claim_in_process,
                        args=(str(self.path), f"sweep:{i}", out))
            for i in range(slots + 3)
        ]
        for w in workers:
            w.start()
        for w in workers:
            w.join(60)
        taken = [g for g in (out.get(timeout=10) for _ in workers) if g]
        self.assertEqual((len(taken), len(set(taken))), (slots, slots))
        held = {e["job"] for e in self.state()["resources"].values()}
        self.assertEqual(len(held), slots)


class TestCli(SlotCase):

    def test_a_quoted_runtime_gets_its_tilde_expanded(self):
        argv = tpu_slots._argv("python3 ~/pkg/framework/runtime.py")
        self.assertEqual(argv[1],
                         os.path.expanduser("~/pkg/framework/runtime.py"))

    def test_state_path_and_exit_codes(self):
        with unittest.mock.patch.dict("os.environ",
                                      {"TPU_AGENT_SUITE_HOME": "/h"},
                                      clear=True):
            self.assertEqual(tpu_slots.state_path("r1"),
                             pathlib.Path("/h/runs/r1/parallel.json"))
            with self.assertRaises(tpu_slots.NeedsInput):
                tpu_slots.state_path(None)
        argv = ["claim", "--holder", "a", "--state", str(self.path)]
        with unittest.mock.patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(tpu_slots.main(argv), tpu_slots.EXIT_NEEDS_INPUT)
        # With --runtime, a claim that finds no file writes it from the
        # session's profiles first.
        with unittest.mock.patch.object(
                tpu_slots, "read_profiles",
                return_value={"v6e-a": PROFILES["v6e-a"]}), \
                unittest.mock.patch.object(
                    tpu_slots, "read_inspect",
                    return_value=lambda _: None), \
                unittest.mock.patch("sys.stdout", new=io.StringIO()) as out:
            self.assertEqual(tpu_slots.main(argv + ["--runtime", "rt"]),
                             tpu_slots.EXIT_OK)
        self.assertEqual(json.loads(out.getvalue())["resource"], "v6e-a")
        argv[2] = "b"
        with unittest.mock.patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(tpu_slots.main(argv), tpu_slots.EXIT_REFUSED)


if __name__ == "__main__":
    unittest.main()
