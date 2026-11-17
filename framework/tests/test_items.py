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
"""Tests for the shared store of delegated work items.

Every test points TPU_AGENT_SUITE_HOME at a temp dir, so nothing here touches
a real ~/.tpu-agent-suite.
"""

from __future__ import annotations

import contextlib
import io
import json
import multiprocessing
import os
import pathlib
import tempfile
import threading
import time
import unittest
from unittest import mock

from framework import items, jsonio, paths, runtime, study, utils

Path = pathlib.Path

RUN_ID = "20260101T000000Z-aaaabbbbcccc"


def _bump(run_id: str, rounds: int) -> None:
    """Queue `rounds` items in a separate process, for the locking test."""
    for _ in range(rounds):
        items.queue(run_id, to="tpu-diagnostics", request={})


class ItemStoreTestCase(unittest.TestCase):
    """Base that isolates the central store into a temp directory."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._prev = os.environ.get("TPU_AGENT_SUITE_HOME")
        os.environ["TPU_AGENT_SUITE_HOME"] = self._tmp.name
        self.home = Path(self._tmp.name)
        self.run_id = RUN_ID
        # The session creates the run directory; the store never does.
        paths.get_run_root(self.run_id).mkdir(parents=True, exist_ok=True)

    def enter_workspace(self, run_id: str) -> None:
        """Work from a workspace whose session recorded `run_id`."""
        workspace = self.home / "ws"
        marker = workspace / paths.TPU_AGENT_SUITE_DIRNAME / paths.ACTIVE_RUN_FILE
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(run_id, encoding="utf-8")
        self.addCleanup(os.chdir, os.getcwd())
        os.chdir(workspace)

    def tearDown(self) -> None:
        if self._prev is None:
            os.environ.pop("TPU_AGENT_SUITE_HOME", None)
        else:
            os.environ["TPU_AGENT_SUITE_HOME"] = self._prev
        self._tmp.cleanup()

    def queue(self, to: str = "tpu-pallas", **kwargs) -> dict:
        return items.queue(self.run_id, to=to, request={}, **kwargs)

    def actions(self, policy: str = "") -> list[tuple[str, str]]:
        """The dispatch list as (action, item id) pairs, in order."""
        plan = items.dispatch(items.load(self.run_id), policy=policy)
        return [(entry["action"], entry["item"]) for entry in plan]

    def backdate(self, item_id: str) -> None:
        """Expire an item's lease without running the sweep."""
        with items.store_lock(self.run_id):
            document = items.load(self.run_id)
            document["items"][item_id]["lease_expires_at"] = (
                "2000-01-01T00:00:00Z")
            items.save(self.run_id, document)

    def stall(self, item_id: str) -> None:
        self.backdate(item_id)
        items.expire_leases(self.run_id)


class QueueTest(ItemStoreTestCase):
    """What `queue` refuses, and that concurrent writers lose nothing."""

    def test_a_run_that_was_never_started_is_refused(self) -> None:
        """A stale id would otherwise build a second store in silence."""
        self.enter_workspace("20200101T000000Z-000000000000")
        for refused in (
                items.resolve_run_id,
                lambda: items.queue("20200101T000000Z-000000000000",
                                    to="tpu-pallas",
                                    request={}),
        ):
            with self.assertRaisesRegex(utils.TpuAgentSuiteError,
                                        "has no directory"):
                refused()
        self.assertFalse(
            (self.home / "runs" / "20200101T000000Z-000000000000").exists())

    def test_the_tightest_depth_cap_wins(self) -> None:
        """A caller's argument can tighten the stored cap, never lift it."""
        for n, (stored, argument) in enumerate(((1, 5), (9, 1), (0, 1))):
            with self.subTest(stored=stored, argument=argument):
                run_id = f"20260101T000000Z-00000000000{n}"
                paths.get_run_root(run_id).mkdir(parents=True)
                items.init(run_id, max_depth=stored)
                parent = items.queue(run_id, to="tpu-pallas", request={})
                with self.assertRaisesRegex(items.ItemError, "the cap is 1"):
                    items.queue(run_id,
                                to="tpu-roofline",
                                request={},
                                parent=parent["id"],
                                max_depth=argument)

    def test_an_agent_already_on_the_chain_is_refused_as_a_cycle(self) -> None:
        parent = self.queue()
        child = self.queue("tpu-critical-review", parent=parent["id"])
        with self.assertRaisesRegex(items.ItemError, "cycle"):
            self.queue(parent=child["id"])

    def test_a_bad_or_unknown_reference_is_refused(self) -> None:
        for ref, message in (("I9#gaps[0]", "no item I9"), ("gaps[0]",
                                                            "I3#gaps")):
            with self.subTest(ref=ref):
                with self.assertRaisesRegex(items.ItemError, message):
                    self.queue(after=[ref])

    def test_concurrent_writers_in_separate_processes_lose_no_item(
            self) -> None:
        items.init(self.run_id)
        workers = [
            multiprocessing.Process(target=_bump, args=(self.run_id, 10))
            for _ in range(4)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=60)
        self.assertEqual(len(items.load(self.run_id)["items"]), 40)


class LifecycleTest(ItemStoreTestCase):
    """Fan-out, partial results, nested waits, refusals and the event log."""

    def test_capacity_and_not_chain_depth_bounds_the_list(self) -> None:
        items.init(self.run_id, capacity=2)
        for _ in range(5):
            self.queue("tpu-diagnostics")
        self.assertEqual(self.actions(), [("start", "I1"), ("start", "I2")])
        items.claim(self.run_id, "I1", owner="w1")
        items.claim(self.run_id, "I2", owner="w2")
        self.assertEqual(self.actions(), [])
        with self.assertRaisesRegex(items.ItemError, "slots are held"):
            items.claim(self.run_id, "I3", owner="w3")
        items.answer(self.run_id, "I1", response={}, owner="w1")
        self.assertEqual(self.actions(), [("start", "I3")])

    def test_a_waiting_item_holds_no_slot_and_no_lease(self) -> None:
        """A lease re-armed mid-wait would let a second worker resume it."""
        items.init(self.run_id, capacity=1)
        self.queue()
        self.queue("tpu-roofline")
        items.claim(self.run_id, "I1", owner="w1")
        self.assertEqual(self.actions(), [])
        items.wait(self.run_id, "I1", on=["I2"], owner="w1")
        self.assertEqual(self.actions(), [("start", "I2")])
        progressed = items.progress(self.run_id,
                                    "I1",
                                    owner="w1",
                                    step="still thinking")
        self.assertEqual(
            (progressed["status"], progressed["lease_expires_at"]),
            (items.STATUS_WAITING, ""))

    def test_a_waiting_item_cannot_resume_into_a_held_slot(self) -> None:
        items.init(self.run_id, capacity=1)
        self.queue()
        self.queue("tpu-roofline")
        self.queue("tpu-diagnostics")
        items.claim(self.run_id, "I1", owner="w1")
        items.wait(self.run_id, "I1", on=["I2"], owner="w1")
        items.claim(self.run_id, "I2", owner="w2")
        items.answer(self.run_id, "I2", response={}, owner="w2")
        items.claim(self.run_id, "I3", owner="w3")
        with self.assertRaisesRegex(items.ItemError, "slots are held"):
            items.claim(self.run_id, "I1", owner="w1")
        self.assertEqual(
            items.load(self.run_id)["items"]["I1"]["status"],
            items.STATUS_WAITING)
        items.answer(self.run_id, "I3", response={}, owner="w3")
        self.assertEqual(
            items.claim(self.run_id, "I1", owner="w1")["status"],
            items.STATUS_CLAIMED)

    def test_a_part_readies_only_its_consumer_and_an_answer_readies_all(
            self) -> None:
        self.queue("tpu-performance-optimization")
        self.queue(after=["I1#gaps[0]"])
        self.queue("tpu-code-arm", after=["I1#gaps[1]"])
        items.claim(self.run_id, "I1", owner="w0")
        items.progress(self.run_id,
                       "I1",
                       owner="w0",
                       part="gaps[0]",
                       value={"op": "ragged_paged_attention"})
        self.assertEqual(self.actions(), [("start", "I2")])
        # The final answer satisfies a part that was never published.
        items.answer(self.run_id, "I1", response={"gaps": []}, owner="w0")
        self.assertEqual(self.actions(), [("start", "I2"), ("start", "I3")])

    def test_a_nested_chain_resumes_every_waiter_in_turn(self) -> None:
        self.queue()
        items.claim(self.run_id, "I1", owner="w1")
        self.queue("tpu-critical-review", parent="I1")
        items.wait(self.run_id, "I1", on=["I2"], owner="w1")
        self.assertEqual(self.actions(), [("start", "I2")])
        items.claim(self.run_id, "I2", owner="w2")
        self.queue("tpu-code-stack", parent="I2")
        items.wait(self.run_id, "I2", on=["I3"], owner="w2")
        self.assertEqual(self.actions(), [("start", "I3")])

        items.claim(self.run_id, "I3", owner="w3")
        items.answer(self.run_id, "I3", response={}, owner="w3")
        self.assertEqual(self.actions(), [("resume", "I2")])
        items.claim(self.run_id, "I2", owner="w2")
        # A failure is still a verdict: the waiter wakes to read it.
        items.fail(self.run_id, "I2", reason="no trace", owner="w2")
        self.assertEqual(self.actions(), [("resume", "I1")])
        items.claim(self.run_id, "I1", owner="w1")
        items.answer(self.run_id, "I1", response={}, owner="w1")
        self.assertEqual(items.open_items(items.load(self.run_id)), [])

    def test_a_wait_that_could_never_wake_is_refused(self) -> None:
        self.queue()
        items.claim(self.run_id, "I1", owner="w1")
        self.queue("tpu-critical-review", after=["I1#gaps[0]"])
        for on, message in (("I1", "cannot wait on itself"),
                            ("I2", "I2 already waits on I1")):
            with self.subTest(on=on):
                with self.assertRaisesRegex(items.ItemError, message):
                    items.wait(self.run_id, "I1", on=[on], owner="w1")

    def test_wrong_owner_and_wrong_state_writes_are_refused(self) -> None:
        self.queue()
        self.queue("tpu-code-arm", after=["I1"])
        items.claim(self.run_id, "I1", owner="w1")
        refusals = (
            ("another worker", "owned by w1",
             lambda: items.progress(self.run_id, "I1", owner="w9", step="nope")
             ),
            ("no owner", "needs the owner",
             lambda: items.answer(self.run_id, "I1", response={})),
            ("held item", "already claimed by w1",
             lambda: items.claim(self.run_id, "I1", owner="w2")),
            ("blocked item", "still waits on I1",
             lambda: items.claim(self.run_id, "I2", owner="w2")),
            ("live claim requeued", "I1 is claimed",
             lambda: items.requeue(self.run_id, "I1")),
        )
        for name, message, write in refusals:
            with self.subTest(name):
                with self.assertRaisesRegex(items.ItemError, message):
                    write()
        items.answer(self.run_id, "I1", response={}, owner="w1")
        for name, write in (
            ("answer",
             lambda: items.answer(self.run_id, "I1", response={}, owner="w1")),
            ("progress",
             lambda: items.progress(self.run_id, "I1", owner="w1", step="x")),
            ("requeue", lambda: items.requeue(self.run_id, "I1")),
        ):
            with self.subTest(f"closed item: {name}"):
                with self.assertRaisesRegex(items.ItemError,
                                            "I1 is completed"):
                    write()

    def test_each_write_appends_one_state_change_with_its_transition(
            self) -> None:
        self.queue()
        items.claim(self.run_id, "I1", owner="w1")
        items.progress(self.run_id,
                       "I1",
                       owner="w1",
                       part="gaps[0]",
                       value={"op": "x"})
        items.progress(self.run_id, "I1",
                       owner="w1")  # renews, records nothing
        items.answer(self.run_id, "I1", response={}, owner="w1")
        log = items.read_events(self.run_id)
        self.assertEqual([record["kind"] for record in log],
                         ["queued", "claimed", "progress", "completed"])
        self.assertEqual(log[2]["part"], "gaps[0]")
        self.assertEqual((log[3]["old_status"], log[3]["new_status"]),
                         ("claimed", "completed"))

    def test_each_policy_picks_its_own_three(self) -> None:
        # I1 d1/p0, I2 d2/p5, I3 d1/p9, I4 d2/p0: every policy differs.
        items.init(self.run_id, capacity=3)
        for lead, lead_priority, child_priority in (("tpu-pallas", 0, 5),
                                                    ("tpu-code-arm", 9, 0)):
            parent = self.queue(lead, priority=lead_priority)
            self.queue("tpu-critical-review",
                       parent=parent["id"],
                       priority=child_priority)
        expected = {
            items.POLICY_BREADTH_FIRST: ["I3", "I1", "I2"],
            items.POLICY_DEPTH_FIRST: ["I2", "I4", "I3"],
            items.POLICY_PRIORITY: ["I3", "I2", "I1"],
        }
        for policy, order in expected.items():
            with self.subTest(policy):
                self.assertEqual(self.actions(policy),
                                 [("start", i) for i in order])
        # With no policy named, the store's own applies.
        items.init(self.run_id, policy=items.POLICY_DEPTH_FIRST)
        self.assertEqual(self.actions(),
                         [("start", i) for i in ["I2", "I4", "I3"]])


class LeaseTest(ItemStoreTestCase):
    """A lease is the only evidence a worker is gone."""

    def setUp(self) -> None:
        super().setUp()
        self.queue()
        items.claim(self.run_id, "I1", owner="w7", lease_seconds=1)

    def test_an_expired_lease_stalls_the_item_until_its_owner_writes(
            self) -> None:
        self.stall("I1")
        self.assertIn("w7", items.load(self.run_id)["items"]["I1"]["failure"])
        # The worker was only slow: a bare progress takes the claim back.
        items.progress(self.run_id, "I1", owner="w7", lease_seconds=3600)
        item = items.load(self.run_id)["items"]["I1"]
        self.assertEqual((item["status"], item["failure"]), ("claimed", ""))
        self.assertEqual(items.read_events(self.run_id)[-1]["kind"], "claimed")
        self.assertEqual(items.expire_leases(self.run_id), [])

    def test_a_stalled_item_is_surfaced_even_with_no_free_slot(self) -> None:
        items.init(self.run_id, capacity=1)
        self.queue("tpu-roofline")
        self.stall("I1")
        items.claim(self.run_id, "I2", owner="w8", lease_seconds=3600)
        self.assertEqual(self.actions(), [("decide", "I1")])

    def test_a_stalled_item_cannot_revive_into_a_held_slot(self) -> None:
        items.init(self.run_id, capacity=1)
        self.queue("tpu-roofline")
        self.stall("I1")
        items.claim(self.run_id, "I2", owner="w8")
        with self.assertRaisesRegex(items.ItemError, "slots are held"):
            items.claim(self.run_id, "I1", owner="w7")
        with self.assertRaisesRegex(items.ItemError, "slots are held"):
            items.progress(self.run_id, "I1", owner="w7")
        self.assertEqual(
            items.load(self.run_id)["items"]["I1"]["status"],
            items.STATUS_STALLED)
        items.answer(self.run_id, "I2", response={}, owner="w8")
        self.assertEqual(
            items.progress(self.run_id, "I1", owner="w7")["status"],
            items.STATUS_CLAIMED)

    def test_only_its_owner_or_a_requeue_can_take_a_stalled_item(self) -> None:
        """Handing it to someone else would run the work twice."""
        self.stall("I1")
        with self.assertRaisesRegex(items.ItemError, "requeue it"):
            items.claim(self.run_id, "I1", owner="w9")
        self.assertEqual(
            items.claim(self.run_id, "I1", owner="w7")["failure"], "")
        self.stall("I1")
        requeued = items.requeue(self.run_id, "I1")
        self.assertEqual((requeued["status"], requeued["owner"]),
                         (items.STATUS_QUEUED, ""))
        self.assertEqual(self.actions(), [("start", "I1")])
        with self.assertRaisesRegex(items.ItemError, "owned by no one"):
            items.answer(self.run_id, "I1", response={}, owner="w7")

    def test_a_late_answer_or_wait_drops_the_stall_reason(self) -> None:
        """A slow worker's result must not still read as failed."""
        self.queue("tpu-roofline")
        items.claim(self.run_id, "I2", owner="w8", lease_seconds=1)
        self.stall("I1")
        self.stall("I2")
        answered = items.answer(self.run_id, "I1", response={}, owner="w7")
        waiting = items.wait(self.run_id, "I2", on=["I1"], owner="w8")
        self.assertEqual((answered["failure"], waiting["failure"]), ("", ""))

    def test_a_requeue_drops_the_discarded_attempts_parts(self) -> None:
        """Otherwise a waiter on a part wakes on the dead worker's output."""
        items.progress(self.run_id, "I1", owner="w7", part="gaps[0]", value={})
        self.queue(after=["I1#gaps[0]"])
        self.stall("I1")
        items.requeue(self.run_id, "I1")
        self.assertEqual(self.actions(), [("start", "I1")])


class DoorbellTest(ItemStoreTestCase):
    """What wakes an idle main agent when someone else writes."""

    def test_an_empty_or_finished_store_is_idle(self) -> None:
        self.assertEqual(
            items.watch(self.run_id, timeout_seconds=30)["reason"], "idle")
        self.queue()
        items.claim(self.run_id, "I1", owner="w1")
        items.answer(self.run_id, "I1", response={}, owner="w1")
        self.assertEqual(
            items.watch(self.run_id, timeout_seconds=30)["reason"], "idle")

    def test_work_in_flight_with_nothing_runnable_times_out(self) -> None:
        self.queue()
        items.claim(self.run_id, "I1", owner="w1")
        started = time.monotonic()
        rung = items.watch(self.run_id, timeout_seconds=0.3, poll_seconds=0.05)
        self.assertEqual(rung["reason"], "timeout")
        self.assertGreaterEqual(time.monotonic() - started, 0.3)

    def test_a_part_published_mid_watch_rings_it(self) -> None:
        self.queue("tpu-performance-optimization")
        self.queue(after=["I1#gaps[0]"])
        items.claim(self.run_id, "I1", owner="w0")
        writer = threading.Timer(0.2,
                                 items.progress,
                                 args=(self.run_id, "I1"),
                                 kwargs={
                                     "owner": "w0",
                                     "part": "gaps[0]",
                                     "value": {}
                                 })
        writer.start()
        try:
            rung = items.watch(self.run_id,
                               timeout_seconds=30,
                               poll_seconds=0.05)
        finally:
            writer.join(timeout=30)
        self.assertEqual([e["item"] for e in rung["runnable"]], ["I2"])

    def test_a_write_in_the_same_mtime_tick_still_rings_it(self) -> None:
        """Coarse mtimes: another process's write must not hide behind one."""
        self.queue("tpu-performance-optimization")
        self.queue(after=["I1#gaps[0]"])
        items.claim(self.run_id, "I1", owner="w0")
        document = items.document_path(self.run_id)
        real_stat = pathlib.Path.stat

        def frozen_stat(path, *args, **kwargs):
            result = real_stat(path, *args, **kwargs)
            if path == document:
                fields = list(result)
                fields[8] = 0  # st_mtime as an int; st_mtime_ns follows it
                result = os.stat_result(fields, {"st_mtime_ns": 0})
            return result

        writer = threading.Timer(0.2,
                                 items.progress,
                                 args=(self.run_id, "I1"),
                                 kwargs={
                                     "owner": "w0",
                                     "part": "gaps[0]",
                                     "value": {}
                                 })
        with mock.patch.object(pathlib.Path, "stat", frozen_stat):
            writer.start()
            try:
                rung = items.watch(self.run_id,
                                   timeout_seconds=5,
                                   poll_seconds=0.05)
            finally:
                writer.join(timeout=30)
        self.assertEqual([e["item"] for e in rung["runnable"]], ["I2"])

    def test_a_dead_worker_rings_it_without_anyone_else_writing(self) -> None:
        self.queue()
        items.claim(self.run_id, "I1", owner="w7", lease_seconds=1)
        self.backdate("I1")
        rung = items.watch(self.run_id, timeout_seconds=30, poll_seconds=0.05)
        self.assertEqual([(e["action"], e["item"]) for e in rung["runnable"]],
                         [("decide", "I1")])


class CliTest(ItemStoreTestCase):
    """`runtime items` reaches the store."""

    def setUp(self) -> None:
        super().setUp()
        self.enter_workspace(self.run_id)

    def cli(self, *argv: str) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(
                io.StringIO()):
            code = runtime.main(["items", *argv])
        return code, out.getvalue()

    def test_a_partial_result_starts_its_consumer_through_the_cli(
            self) -> None:
        steps = [
            ("queue", "--to", "tpu-performance-optimization", "--request",
             '{"top_n": 3}'),
            ("queue", "--to", "tpu-pallas", "--request", "{}", "--after",
             "I1#gaps[0]"),
            ("claim", "I1", "--owner", "w0"),
            ("progress", "I1", "--owner", "w0", "--part", "gaps[0]", "--value",
             '{"op": "rpa"}'),
        ]
        for argv in steps:
            self.assertEqual(self.cli(*argv)[0], 0, argv)
        code, out = self.cli("dispatch")
        self.assertEqual([(e["action"], e["item"]) for e in json.loads(out)],
                         [("start", "I2")])
        self.assertEqual(
            items.load(self.run_id)["items"]["I1"]["parts"]["gaps[0]"],
            {"op": "rpa"})

    def test_an_answer_with_no_response_is_refused(self) -> None:
        """It used to close the item with a `null` result."""
        self.queue()
        items.claim(self.run_id, "I1", owner="w1")
        self.assertNotEqual(self.cli("answer", "I1", "--owner", "w1")[0], 0)
        self.assertEqual(
            items.load(self.run_id)["items"]["I1"]["status"],
            items.STATUS_CLAIMED)

    def test_dispatch_stalls_an_expired_claim_first(self) -> None:
        """There is no separate `expire` verb to remember to run."""
        self.queue()
        items.claim(self.run_id, "I1", owner="w7", lease_seconds=1)
        self.backdate("I1")
        code, out = self.cli("dispatch")
        self.assertEqual([(e["action"], e["item"]) for e in json.loads(out)],
                         [("decide", "I1")])


class AdmissionTest(ItemStoreTestCase):
    """`queue` and `answer` held to the package's published graph and schemas."""

    PACKAGE = "probe-pkg@1.0.0"

    def setUp(self) -> None:
        super().setUp()
        self.root = self.home / paths.PACKAGES_DIRNAME / self.PACKAGE
        (self.root / "schemas").mkdir(parents=True)
        jsonio.write_json(self.root / "schemas" / "gap.json", {
            "type": "object",
            "required": ["gap"],
        })
        jsonio.write_json(self.root / "schemas" / "result.json", {
            "type": "object",
            "properties": {
                "artifacts": {
                    "type": "array"
                }
            },
        })
        jsonio.write_json(
            self.root / "tpu-agent-suite-build.json", {
                "contracts": {
                    "tpu-pallas": {
                        "accepts": "schemas/gap.json",
                        "produces": "schemas/result.json",
                    },
                },
                "delegation": {
                    "graph": {
                        "tpu-agent": ["tpu-pallas", "tpu-critical-review"],
                        "tpu-pallas": ["tpu-critical-review"],
                        "tpu-critical-review": ["tpu-code-stack"],
                        "tpu-code-stack": [],
                    },
                    "roots": ["tpu-critical-review", "tpu-pallas"],
                    "max_depth": 3,
                },
            })

    def admit(self, to: str, **kwargs) -> dict:
        kwargs.setdefault("request", {"gap": "x"})
        return items.queue(self.run_id, to=to, package=self.PACKAGE, **kwargs)

    def test_a_request_is_held_to_its_real_requesters_edges(self) -> None:
        review = self.admit("tpu-critical-review")
        builtin = self.admit("general-purpose", request={})
        refused = (
            ("tpu-pallas", review["id"], "tpu-agent"),  # the parent wins
            ("tpu-code-stack", "", "tpu-critical-review"),  # no parent: roots
            (" tpu-code-stack", "", ""),  # a padded name is the same name
            ("tpu-code-stack", builtin["id"], ""),  # a built-in is no hop
        )
        for to, parent, claimed in refused:
            with self.subTest(to=to, parent=parent, requested_by=claimed):
                with self.assertRaises(items.ItemError):
                    self.admit(to,
                               request={},
                               parent=parent,
                               requested_by=claimed)
        self.admit("tpu-pallas", parent=builtin["id"])

    def test_a_caller_cannot_raise_the_published_cap(self) -> None:
        parent = self.admit("tpu-pallas")
        review = self.admit("tpu-critical-review", parent=parent["id"])
        stack = self.admit("tpu-code-stack", parent=review["id"])
        with self.assertRaisesRegex(items.ItemError, "the cap is 3"):
            self.admit("tpu-knowledge-base",
                       request={},
                       parent=stack["id"],
                       max_depth=99)

    def test_requests_and_answers_meet_the_callees_contract(self) -> None:
        with self.assertRaisesRegex(items.ItemError, "request to tpu-pallas"):
            self.admit("tpu-pallas", request={"wrong": 1})
        self.admit("tpu-pallas")
        items.claim(self.run_id, "I1", owner="w1")
        # The SubagentStop rules, not just the schema.
        with self.assertRaisesRegex(items.ItemError, "at least one artifact"):
            items.answer(self.run_id,
                         "I1",
                         response={
                             "status": "completed",
                             "artifacts": []
                         },
                         owner="w1",
                         package=self.PACKAGE)

    def test_the_cli_takes_the_package_from_the_run_not_a_flag(self) -> None:
        jsonio.write_json(
            paths.get_run_root(self.run_id) / "run.json", {
                "run_id": self.run_id,
                "package": self.PACKAGE
            })
        self.enter_workspace(self.run_id)
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            code = runtime.main([
                "items", "queue", "--to", "tpu-pallas", "--request",
                '{"wrong": 1}'
            ])
        self.assertNotEqual(code, 0)
        self.assertEqual(items.load(self.run_id)["items"], {})

    def test_a_package_built_before_the_graph_still_runs(self) -> None:
        build = jsonio.load_json(self.root / "tpu-agent-suite-build.json")
        del build["delegation"]
        jsonio.write_json(self.root / "tpu-agent-suite-build.json", build)
        review = self.admit("tpu-critical-review")
        self.assertEqual(
            self.admit("tpu-pallas", parent=review["id"])["depth"], 2)


class StudyTimelineTest(ItemStoreTestCase):

    def test_changes_reach_the_attached_study_and_never_fail_on_it(
            self) -> None:
        study.start_study(study_id="probe-study",
                          protocol={
                              "primary_metric": {
                                  "name": "tokens_per_s",
                                  "unit": "tok/s",
                                  "direction": "maximize",
                              },
                              "measurement_cell": {
                                  "topology": "v6e-2x2x1"
                              },
                              "replicate_count": 3,
                              "noise_floor_pct": 0.4,
                              "stopping_rule": {
                                  "max_rounds": 10
                              },
                              "admissibility_gates": [],
                              "mode": "approve",
                          },
                          package="probe-pkg",
                          target="probe-model",
                          owner="tester")
        study.attach_run_to_study("probe-study", self.run_id)
        self.queue()
        items.claim(self.run_id, "I1", owner="w1")
        mirrored = [
            event for event in study.read_study_events("probe-study")
            if event.get("kind") == "items"
        ]
        self.assertEqual([(e["change"], e["item"]) for e in mirrored],
                         [("queued", "I1"), ("claimed", "I1")])
        # Best-effort: an unwritable study never fails a delegation.
        with mock.patch.object(study,
                               "append_study_event",
                               side_effect=OSError("disk full")):
            self.assertEqual(self.queue()["status"], items.STATUS_QUEUED)


if __name__ == "__main__":
    unittest.main()
