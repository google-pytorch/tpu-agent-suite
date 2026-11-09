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
"""Tests for gke_job inspect (node pools, preflight) and multi-host jobs."""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import unittest
from typing import Any
from unittest.mock import patch

# Imports below follow this sys.path setup, so the file runs directly.
# ruff: noqa: E402
REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework.compute.drivers import (auth_utils, base, gcs_staging, gke_job,
                                       jobs)
from framework.tests.compute.gke_job.helpers import (FakeKubectl, failed_pod,
                                                     hello_script, job_store,
                                                     make_driver,
                                                     set_up_module)


def setUpModule():  # pylint: disable=invalid-name
    set_up_module()


def node_json(count: int = 1,
              chips: int = 4,
              accelerator: str = "tpu7x") -> str:
    labels = {
        base.ACCELERATOR_LABEL: accelerator,
        base.TOPOLOGY_LABEL: "2x2x1"
    }
    return json.dumps({
        "items": [
            dict(metadata=dict(name=f"node-{i}", labels=labels),
                 status=dict(allocatable={base.TPU_RESOURCE: str(chips)}))
            for i in range(count)
        ]
    })


def pods_json(*requests: tuple[str, int, str]) -> str:
    """Pods as `kubectl get pods` lists them; a Pending one is unscheduled."""
    pods = []
    for name, chips, phase in requests:
        pod = failed_pod(phase=phase,
                         name=name,
                         node=None if phase == "Pending" else "gke-tpu-node-1")
        pod["metadata"]["namespace"] = "default"
        pod.setdefault("spec", {})["containers"] = [
            dict(resources=dict(limits={base.TPU_RESOURCE: str(chips)}))
        ]
        pods.append(pod)
    return json.dumps({"items": pods})


class TestGkeJobInspect(unittest.TestCase):

    def inspect(self, fake: FakeKubectl, **overrides: Any):
        driver = make_driver(**overrides)
        with patch.object(subprocess, "run", fake):
            return driver.inspect()

    def _preflight(self, responses, **overrides):
        """Preflight against a cluster that answers, unless told otherwise.

        `responses` are matched first, so a narrower key can answer one of
        several calls a default would.
        """
        defaults = {
            "config current-context": (0, "ctx", ""),
            "version": (0, '"serverVersion"', ""),
            "auth can-i": (0, "yes\n", ""),
            "get nodes": (0, "node/tpu-1\n", ""),
            "auth list": (0, "me@example.com\n", ""),
        }
        fake = FakeKubectl({
            **responses,
            **{
                k: v
                for k, v in defaults.items() if k not in responses
            }
        })
        env = {
            k: v
            for k, v in os.environ.items()
            if k != base.GAC_ENV and not k.startswith("GKE_")
        }
        env.update(overrides.pop("env", {}))
        with patch.object(shutil, "which",
                          return_value="/bin/tool"), \
                patch.dict(os.environ, env, clear=True), \
                patch.object(subprocess, "run", fake):
            report = make_driver(**overrides).preflight()
        return report, {c.name: c for c in report.checks}, fake

    def test_preflight_checks_the_image_in_artifact_registry(self):
        image = "us-docker.pkg.dev/p/repo/torch-tpu:v1"
        _, checks, fake = self._preflight(
            {"images describe": (0, "sha256:abc\n", "")}, image=image)
        self.assertEqual(checks["image"].status, "ok")
        self.assertIn(image, " ".join(fake.find("images describe")))
        # What gcloud really prints for a missing tag or digest: the error,
        # then a usage block whose last line says nothing is wrong.
        real_not_found = (
            "ERROR: (gcloud.artifacts.docker.images.describe) Image not "
            "found.\n\nA valid container image can be referenced by tag or "
            "digest, has the format of\n"
            "  LOCATION-docker.pkg.dev/PROJECT-ID/REPOSITORY-ID/IMAGE:tag\n"
            "  LOCATION-docker.pkg.dev/PROJECT-ID/REPOSITORY-ID/IMAGE@sha256:"
            "digest\n")
        _, checks, _ = self._preflight(
            {"images describe": (1, "", real_not_found)}, image=image)
        self.assertEqual(checks["image"].status, "fail")
        self.assertIn("Image not found", checks["image"].detail)
        self.assertIn(
            "gcloud artifacts docker images list "
            "us-docker.pkg.dev/p/repo/torch-tpu", checks["image"].fix)
        self.assertNotIn("sha256:digest", checks["image"].detail)
        # Not allowed to read it: the nodes may still pull it.
        _, checks, _ = self._preflight(
            {"images describe": (1, "", "PERMISSION_DENIED: denied")},
            image=image)
        self.assertEqual(checks["image"].status, "warn")
        # Elsewhere, it cannot be checked without pulling it.
        _, checks, fake = self._preflight({}, image="docker.io/me/img:1")
        self.assertEqual(checks["image"].status, "warn")
        self.assertIsNone(fake.find("images describe"))

    CONTEXT = "gke_tpu-nexus-767207_us-central2-b_tpu-nexus-cluster"
    NODE_SA = "141197224043-compute@developer.gserviceaccount.com"

    @staticmethod
    def _readers(member):
        """A repository policy granting `member` roles/artifactregistry.reader."""
        return {
            "bindings": [{
                "role": "roles/artifactregistry.reader",
                "members": [member]
            }]
        }

    def _node_pull(self, image, policy=None, driver=None, **responses):
        """Preflight of `image` on a gke_* context, with the TPU nodes in
        pool 'tpu-pool' running as the default Compute Engine account."""
        pools = [{
            "name": "tpu-pool",
            "config": {
                "serviceAccount": "default"
            }
        }, {
            "name": "other-pool",
            "config": {
                "serviceAccount": "other@x.iam.gserviceaccount.com"
            }
        }]
        _, checks, fake = self._preflight(
            {
                "config current-context": (0, self.CONTEXT, ""),
                "images describe": (0, "sha256:abc\n", ""),
                "get nodes": (0, "tpu-pool\ntpu-pool\n", ""),
                "node-pools list": (0, json.dumps(pools), ""),
                "projects describe":
                (0, json.dumps({"projectNumber": "141197224043"}), ""),
                "get-iam-policy":
                ((0, json.dumps(policy), "") if policy is not None else
                 (1, "", "ERROR: PERMISSION_DENIED")),
                **responses
            },
            image=image,
            **(driver or {}))
        return checks, fake

    def test_preflight_warns_the_nodes_may_not_pull_a_cross_project_image(
            self):
        """The host can read the image; the nodes pull as themselves."""
        image = ("us-docker.pkg.dev/ml-oss-artifacts-transient/"
                 "torch-tpu-docker-container/torch-tpu:nightly-latest")
        checks, fake = self._node_pull(image)
        pull = checks["image:node_pull"]
        self.assertEqual(pull.status, "warn")
        # Only the pool the TPU nodes are in, with "default" resolved.
        self.assertIn(self.NODE_SA, pull.detail)
        self.assertNotIn("other@x", pull.detail)
        self.assertIn("ml-oss-artifacts-transient", pull.detail)
        self.assertIn("cannot read its policy", pull.detail)
        self.assertIn(
            "add-iam-policy-binding torch-tpu-docker-container --location us "
            "--project ml-oss-artifacts-transient --role "
            f"roles/artifactregistry.reader --member serviceAccount:"
            f"{self.NODE_SA}", pull.fix)
        self.assertIn("--cluster tpu-nexus-cluster --location us-central2-b",
                      " ".join(fake.find("node-pools list")))
        # A policy that leaves them out says so.
        checks, _ = self._node_pull(image,
                                    policy=self._readers("group:someone@x"))
        self.assertIn("does not name them", checks["image:node_pull"].detail)
        # One that grants them, or everyone, read: ok.
        for member in (f"serviceAccount:{self.NODE_SA}", "allUsers"):
            checks, _ = self._node_pull(image, policy=self._readers(member))
            self.assertEqual(checks["image:node_pull"].status, "ok", member)

    def test_preflight_node_pull_picks_a_scaled_down_pool_by_topology(self):
        """No TPU nodes to name the pool: the pool of that topology, or with
        none of that topology, a generic account rather than every pool's."""
        pools = json.dumps([{
            "placementPolicy": {
                "tpuTopology": "2x2x1"
            },
            "config": {
                "serviceAccount": "tpu@x"
            }
        }, {
            "config": {
                "serviceAccount": "cpu@x"
            }
        }])
        for topology, said in (("2x2x1", "tpu@x"),
                               ("4x4", "the node pools' service account")):
            checks, _ = self._node_pull("us-docker.pkg.dev/p/repo/img:1",
                                        driver={"tpu_topology": topology},
                                        **{
                                            "get nodes": (0, "", ""),
                                            "node-pools list": (0, pools, "")
                                        })
            self.assertIn(said, checks["image:node_pull"].detail)
            self.assertNotIn("cpu@x", checks["image:node_pull"].detail)

    def test_preflight_reachability(self):
        """A cluster that never answers is not a credentials problem, and
        nothing is asked of it; one that answers flags only what is unset."""
        report, checks, _ = self._preflight({
            "version": (1, "", "Unable to connect to the server: context "
                        "deadline exceeded")
        })
        self.assertFalse(report.reachable)
        self.assertEqual(checks["cluster"].status, "fail")
        self.assertNotIn("rbac:create jobs", checks)
        self.assertIn("authorized networks", checks["cluster"].fix)
        self.assertIn("--dns-endpoint", checks["cluster"].fix)
        _, checks, _ = self._preflight(
            {"version": (1, "", "error: You must be logged in")})
        self.assertNotIn("authorized networks", checks["cluster"].fix)
        report, checks, _ = self._preflight({}, image="")
        self.assertTrue(report.reachable)
        self.assertEqual(
            {name
             for name, c in checks.items() if c.status != "ok"},
            {"dispatch_settings"})

    def test_preflight_warns_on_an_unknown_accelerator(self):
        report, checks, fake = self._preflight({"get nodes": (0, "", "")},
                                               tpu_accelerator='pu-v6e-slice"')
        self.assertEqual(checks["tpu_accelerator"].status, "warn")
        self.assertIn("tpu-v6e-slice", checks["tpu_accelerator"].fix)
        # No node yet may be a pool that scales from zero: never a failure.
        self.assertEqual(checks["tpu_nodes"].status, "warn")
        self.assertIn("scales from zero", checks["tpu_nodes"].fix)
        self.assertIn('pu-v6e-slice"', " ".join(fake.find("get nodes")))
        self.assertNotEqual(report.status, "fail")
        _, checks, _ = self._preflight({})
        self.assertNotIn("tpu_accelerator", checks)
        self.assertEqual(checks["tpu_nodes"].status, "ok")

    def test_preflight_checks_the_namespace_rights_and_kueue(self):
        _, checks, fake = self._preflight(
            {
                "get namespace": (1, "", 'Error from server (Forbidden): '
                                  'namespaces "ns" is forbidden'),
                "get localqueue": (1, "", "error: the server doesn't have a "
                                   'resource type "localqueue"'),
                "get workloadpriorityclass": (0, "low", ""),
            },
            namespace="ns",
            queue="q",
            priority="low")
        self.assertEqual(checks["namespace"].status, "warn")
        self.assertEqual(checks["rbac:create jobs"].status, "ok")
        self.assertEqual(checks["rbac:create configmaps"].status, "ok")
        self.assertEqual(checks["kueue:queue"].status, "fail")
        self.assertIn("Install Kueue", checks["kueue:queue"].fix)
        self.assertEqual(checks["kueue:priority"].status, "ok")
        self.assertIn("-n ns", " ".join(fake.find("get localqueue")))
        self.assertIn("-n ns", " ".join(fake.find("auth can-i")))

    def test_preflight_fails_an_account_that_may_not_create_jobs(self):
        _, checks, _ = self._preflight({"auth can-i": (1, "no\n", "")})
        self.assertEqual(checks["rbac:create jobs"].status, "fail")
        self.assertIn("Ask a cluster admin", checks["rbac:create jobs"].fix)

    def test_preflight_checks_staging_only_with_a_bucket(self):
        staged = [base.PreflightCheck("staging:signing", "fail", "no")]
        with patch.object(gcs_staging, "preflight_checks",
                          return_value=staged) as staging:
            _, checks, _ = self._preflight({}, staging_bucket="gs://b/")
            staging.assert_called_once_with("b",
                                            credentials=None,
                                            pod_service_account=None)
            self.assertEqual(checks["staging:signing"].status, "fail")
            # Setup asks again for what lets the pod reach the bucket.
            self.assertEqual(checks["staging:signing"].fields,
                             list(gcs_staging.POD_UPLOAD_FIELDS))
            self._preflight({})
            staging.assert_called_once()

    def test_preflight_names_tpu_pools_without_workload_identity(self):
        _, checks, _ = self._preflight(
            {
                f"{gke_job.gke_common.WORKLOAD_IDENTITY_LABEL}=true":
                (0, "", ""),
                "get nodes": (0, "tpu-pool\n", ""),
            },
            pod_service_account="ksa")
        self.assertEqual(checks["tpu_nodes"].status, "warn")
        self.assertIn("tpu-pool", checks["tpu_nodes"].detail)
        self.assertIn("GKE_METADATA", checks["tpu_nodes"].fix)

    @staticmethod
    def _pvc(phase: str = "Bound",
             modes: tuple[str, ...] = ("ReadWriteMany", ),
             storage_class: str = "filestore-rwx") -> tuple[int, str, str]:
        status: dict[str, Any] = {"phase": phase}
        if phase == "Bound":
            status.update(accessModes=list(modes), capacity={"storage": "1Ti"})
        return (0,
                json.dumps({
                    "spec": {
                        "accessModes": list(modes),
                        "storageClassName": storage_class
                    },
                    "status": status
                }), "")

    def test_preflight_checks_the_hf_cache_pvc_only_when_set(self):
        _, checks, fake = self._preflight({})
        self.assertNotIn("hf_cache_pvc", checks)
        self.assertIsNone(fake.find("get pvc"))

        _, checks, fake = self._preflight({"get pvc": self._pvc()},
                                          namespace="ns",
                                          hf_cache_pvc="hf-cache")
        self.assertEqual(checks["hf_cache_pvc"].status, "ok")
        self.assertIn("Bound, ReadWriteMany, 1Ti",
                      checks["hf_cache_pvc"].detail)
        self.assertIn("-n ns", " ".join(fake.find("get pvc hf-cache")))

    def test_preflight_judges_an_unusable_hf_cache_pvc(self):
        # A missing claim leaves every Job naming it Pending, so the
        # installer must flag it before anything is written.
        pending = self._pvc(phase="Pending")
        forbidden = (1, "", "Error from server (Forbidden)")
        cases = [
            # (the claim, its storage class's binding mode, status, said)
            ((1, "", 'Error from server (NotFound): persistentvolumeclaims '
              '"hf-cache" not found'), None, "fail",
             "Ask a cluster admin to create a ReadWriteMany"),
            (self._pvc(modes=("ReadWriteOnce", )), None, "fail",
             "ReadWriteOnce, not ReadWriteMany"),
            # Unreadable is not proof it is missing.
            (forbidden, None, "warn", ""),
            # Pending is healthy under a class that binds on first use, a
            # failure under one that should already have bound it, and
            # undecided under one this account cannot read.
            (pending, (0, "WaitForFirstConsumer", ""), "warn",
             "until the first Job"),
            (pending, (0, "Immediate", ""), "fail", "not Bound"),
            (pending, forbidden, "warn", ""),
        ]
        for answer, binding, status, said in cases:
            with self.subTest(status=status, said=said):
                report, checks, fake = self._preflight(
                    {
                        "get pvc": answer,
                        "get storageclass": binding or (0, "", "")
                    },
                    hf_cache_pvc="hf-cache")
                check = checks["hf_cache_pvc"]
                self.assertEqual(check.status, status)
                self.assertIn(said, check.detail + check.fix)
                if status == "fail":
                    self.assertEqual(report.status, "fail")
                if binding:
                    self.assertIn("filestore-rwx",
                                  " ".join(fake.find("get storageclass")))

    def test_a_pool_with_no_node_still_reports_the_profiles_tpu(self):
        """Review on #840: an autoscaled pool at zero nodes read as
        `accelerator: none`, leaving a CPU preflight Job nothing to plan for
        and TP nothing to be sized to. What the profile names is reported in
        `details`, and the top-level fields still say no node is there."""
        fake = FakeKubectl({"get nodes": (0, json.dumps({"items": []}), "")})
        with patch.object(gke_job.manifest,
                          "_chip_spec",
                          return_value=({
                              "hbm_gb": 192.0,
                              "devices_per_chip": 2
                          }, "")):
            profile = self.inspect(fake, tpu_accelerator="tpu7x")
        self.assertEqual((profile.status, profile.accelerator, profile.chips),
                         ("no_accelerator", "none", 0))
        details = profile.details
        self.assertEqual((details["tpu_accelerator"], details["job_devices"],
                          details["hbm_gb_per_device"]), ("tpu7x", 8, 96.0))

    def test_job_devices_and_hbm_come_from_tpu_info(self):
        # A 2x2x1 tpu7x Job is 4 chips but 8 devices of 96 GB each; reporting
        # only chips and 192 GB per chip is what led a run to TP=4 on half a
        # slice. What tpu_info cannot tell is left out rather than guessed.
        cases = [
            # (label, tpu_info importable, hbm_gb, devices_per_chip, note)
            ("tpu7x", True, 192.0, 2, None),
            ("tpu-v99-slice", True, 0.0, None, "tpu-v99-slice"),
            ("tpu7x", False, 0.0, None,
             "Before anything else, ask the user to run `"),
        ]
        for label, has_tpu_info, hbm_gb, per_chip, note in cases:
            with self.subTest(label=label, tpu_info=has_tpu_info):
                fake = FakeKubectl({
                    "get nodes": (0, node_json(accelerator=label), ""),
                    "get pods": (0, pods_json(), ""),
                })
                missing = {} if has_tpu_info else {
                    "tpu_info": None,
                    "tpu_info.device": None
                }
                with patch.dict(sys.modules, missing):
                    profile = self.inspect(fake, tpu_accelerator=label)
                details = profile.details
                self.assertEqual(profile.hbm_gb, hbm_gb)
                self.assertEqual(details["job_chips"], 4)
                self.assertEqual(details.get("devices_per_chip"), per_chip)
                if per_chip:
                    self.assertEqual(details["job_devices"], 4 * per_chip)
                    self.assertEqual(details["hbm_gb_per_device"],
                                     hbm_gb / per_chip)
                else:
                    self.assertNotIn("job_devices", details)
                    self.assertIn(note, details["hbm_gb_note"])

    def test_running_pods_subtract_from_available_chips(self):
        fake = FakeKubectl({
            "get nodes": (0, node_json(count=2, chips=4), ""),
            "get pods": (0, pods_json(("busy", 4, "Running")), ""),
        })
        profile = self.inspect(fake)
        self.assertEqual((profile.chips, profile.available_chips), (8, 4))
        self.assertEqual(profile.held_by, ["default/busy"])
        # Only the configured pool's nodes are counted.
        selector = " ".join(fake.find("get nodes"))
        self.assertIn(f"{base.ACCELERATOR_LABEL}=tpu7x", selector)
        self.assertIn(f"{base.TOPOLOGY_LABEL}=2x2x1", selector)
        fake.responses["get nodes"] = (0, node_json(count=1, chips=4), "")
        self.assertEqual(self.inspect(fake).status, "busy")

    def test_finished_and_unscheduled_pods_do_not_hold_chips(self):
        # A Pending pod with no node is waiting for chips, not holding them;
        # counted, it read an empty cluster as busy.
        fake = FakeKubectl({
            "get nodes": (0, node_json(count=1, chips=4), ""),
            "get pods":
            (0, pods_json(("old", 4, "Succeeded"),
                          ("waiting", 4, "Pending")), ""),
        })
        profile = self.inspect(fake)
        self.assertEqual(profile.available_chips, 4)
        self.assertEqual(profile.held_by, [])

    def test_unlistable_pods_leave_contention_unknown(self):
        # Not 0 and not `chips`: one would report a healthy pool as saturated
        # and the other would send a scheduler at chips somebody else holds.
        fake = FakeKubectl({
            "get nodes": (0, node_json(count=1, chips=4), ""),
            "get pods": (1, "", "Forbidden"),
        })
        profile = self.inspect(fake)
        self.assertIsNone(profile.available_chips)
        self.assertEqual(profile.status, "unknown_contention")


def _indexed_pod(index: int,
                 phase: str = "Running",
                 node: str | None = "node",
                 exit_code: int | None = None,
                 finished: str = "") -> dict[str, Any]:
    """A pod of an Indexed Job, as `kubectl get pods -o json` lists it."""
    status: dict[str, Any] = {"phase": phase}
    if exit_code is not None:
        terminated = dict(exitCode=exit_code, finishedAt=finished)
        status["containerStatuses"] = [dict(state=dict(terminated=terminated))]
    labels = {gke_job.gke_common.COMPLETION_INDEX_KEY: str(index)}
    return dict(metadata=dict(name=f"run-1-{index}", labels=labels),
                status=status,
                spec=dict(nodeName=node) if node else {})


class TestGkeJobMultiHost(unittest.TestCase):
    """A slice larger than one node: one pod per node, the leader read."""

    def test_hosts_from_the_topology(self):
        for accelerator, total, want in (
            ("tpu7x", 4, (1, 4)),
            ("tpu7x", 8, (2, 4)),
            ("tpu-v6e-slice", 16, (4, 4)),
            ("tpu-v99", 8, (1, 8)),  # unknown: as before
        ):
            self.assertEqual(gke_job.manifest.hosts_for(accelerator, total),
                             want)
        details: dict[str, Any] = {}
        gke_job.manifest.job_sizing(
            make_driver(tpu_topology="2x2x2").config, "unknown", details)
        self.assertEqual((details["job_chips"], details["job_hosts"]), (8, 2))
        for overrides, want in (({}, (4, 2)), ({"tpu_chips": 8}, (8, 1))):
            d = make_driver(tpu_accelerator="tpu-v6e-slice",
                            tpu_topology="2x4",
                            **overrides)
            with patch.object(
                    subprocess, "run",
                    FakeKubectl({"get nodes": (0, node_json(chips=4), "")})):
                gke_job.manifest.fit_to_nodes(d.kube, d.config)
            self.assertEqual((d.config.tpu_chips, d.config.tpu_hosts), want)

    def test_submit_builds_an_indexed_job_behind_a_service(self):
        fake = FakeKubectl({"apply": (0, "uid-1", "")})
        script = hello_script(self)
        with job_store(), patch.object(subprocess, "run", fake):
            make_driver(tpu_topology="2x2x2",
                        namespace="ns").submit(str(script),
                                               job_name="run-1",
                                               serve=True)
            self.assertEqual(jobs.JobRecord.load("run-1").extra["hosts"], 2)
        spec = fake.applied_manifest()["spec"]
        pod_spec = spec["template"]["spec"]
        container = pod_spec["containers"][0]
        env = {
            e["name"]: e.get("value", e.get("valueFrom"))
            for e in container["env"]
        }
        self.assertEqual(
            (spec["completionMode"], spec["completions"], spec["parallelism"],
             spec["backoffLimit"], pod_spec["restartPolicy"],
             pod_spec["subdomain"], container["resources"]["limits"]),
            ("Indexed", 2, 2, 0, "Never", "run-1", {
                base.TPU_RESOURCE: 4
            }))
        self.assertEqual((env["TPU_NUM_HOSTS"], env["TPU_WORKER_HOSTNAMES"]),
                         ("2", "run-1-0.run-1.ns.svc,run-1-1.run-1.ns.svc"))
        service = next(
            json.loads(s) for s in fake.stdins
            if s and json.loads(s)["kind"] == "Service")
        self.assertIs(service["spec"]["publishNotReadyAddresses"], True)
        anti = pod_spec["affinity"]["podAntiAffinity"][
            "requiredDuringSchedulingIgnoredDuringExecution"][0]
        self.assertEqual(anti["topologyKey"],
                         gke_job.gke_common.NODEPOOL_LABEL)
        # For hosts > 1, a refused Service fails submit and cleans up the Job.
        refuse = FakeKubectl({
            "apply -f - -o jsonpath": (0, "uid-1", ""),
            "apply -f -": (1, "", "Error from server (Forbidden): services"),
        })
        with job_store(), patch.object(subprocess, "run", refuse), \
                patch.object(auth_utils, "adc_token",
                             return_value=("", "no ADC")):
            with self.assertRaises(base.ComputeConfigError):
                make_driver(tpu_topology="2x2x2").submit(str(script),
                                                         job_name="run-2")
            self.assertIsNone(jobs.JobRecord.load("run-2"))
        self.assertIsNotNone(refuse.find("delete job run-2"))
        self.assertIsNotNone(refuse.find("delete configmap run-2"))

    def test_status_reads_every_host(self):

        def check(job_st: dict[str, Any], pods: list[dict[str, Any]]):
            job = dict(spec=dict(completionMode="Indexed", completions=2),
                       status=job_st)
            fake = FakeKubectl({
                "get job": (0, json.dumps(job), ""),
                "get pods": (0, json.dumps({"items": pods}), ""),
            })
            with job_store(), patch.object(subprocess, "run", fake):
                return make_driver(tpu_topology="2x2x2").get_status("run-1")

        waiting = check(
            {"active": 2},
            [_indexed_pod(0),
             _indexed_pod(1, "Pending", node=None)])
        self.assertEqual((waiting.status, waiting.details["lagging_host"],
                          waiting.details["pending_reason"]),
                         ("PENDING", 1, base.PENDING_UNSCHEDULED))
        running = check({"active": 2}, [_indexed_pod(0), _indexed_pod(1)])
        self.assertEqual((running.status, running.details["pod_name"]),
                         ("RUNNING", "run-1-0"))
        self.assertEqual(
            check({
                "active": 1,
                "succeeded": 1
            }, [_indexed_pod(0, "Succeeded", exit_code=0),
                _indexed_pod(1)]).status, "RUNNING")
        failed = check({"failed": 1}, [
            _indexed_pod(0, "Failed", exit_code=143, finished="t9"),
            _indexed_pod(1, "Failed", exit_code=1, finished="t1"),
        ])
        self.assertEqual((failed.status, failed.exit_code), ("FAILED", 1))
        self.assertTrue(failed.error.startswith("host 1 of 2: "), failed.error)

    def test_leader_owns_logs_port_forward_and_outputs(self):
        pods = json.dumps({"items": [_indexed_pod(1), _indexed_pod(0)]})
        fake = FakeKubectl({"get pods": (0, pods, "")})
        driver = make_driver(tpu_topology="2x2x2")
        with job_store(), patch.object(subprocess, "run", fake), \
                patch.object(auth_utils, "resolve_credentials"):
            jobs.JobRecord.create(job_id="run-1",
                                  backend="gke_job",
                                  extra={"hosts": 2})
            gke_job.log_retrieval.pod_logs(driver.kube, "run-1")
            gke_job.log_retrieval.pod_logs(driver.kube, "run-1", [
                _indexed_pod(0, "Failed", exit_code=143, finished="t9"),
                _indexed_pod(1, "Failed", exit_code=1, finished="t1"),
            ])
            cmd, _ = driver._port_forward_command(8000, 8000, "run-1")
        self.assertIn("pod/run-1-0", fake.find(" logs "))
        self.assertIn("pod/run-1-1", fake.calls[-2])
        self.assertIn("pod/run-1-0", cmd)
        bootstrap = gcs_staging.build_pod_bootstrap_command(
            script_mount="/m",
            script_name="w.sh",
            interpreter="bash",
            logged_outputs=["out.txt"],
            job="run-1")[2]
        self.assertIn(
            f"! {gcs_staging.NOT_LEADER} || exit $WORKLOAD_EXIT_CODE",
            bootstrap)


if __name__ == "__main__":
    unittest.main()
