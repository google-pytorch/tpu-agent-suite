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
"""Tests for gke_job parameters, job names and submission."""

from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
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

from framework.compute import ComputeConfigError
from framework.compute.drivers import auth_utils, base, gke_job, jobs
from framework.tests.compute.gke_job.helpers import (FakeKubectl, hello_script,
                                                     job_store, make_driver,
                                                     set_up_module)


def setUpModule():  # pylint: disable=invalid-name
    set_up_module()


def env_behind(prologue: list[str], image_env: dict[str, str],
               names: Any) -> dict[str, str]:
    """Which of `names` a workload behind the cache `prologue` sees set.

    Returns each one's value, in a container whose image sets `image_env`.
    """
    printer = ("import json, os, sys; print(json.dumps({n: os.environ[n] "
               "for n in sys.argv[1:] if n in os.environ}))")
    proc = subprocess.run(prologue + [sys.executable, "-c", printer, *names],
                          env={
                              "PATH": os.environ["PATH"],
                              **image_env
                          },
                          capture_output=True,
                          text=True,
                          check=True)
    return json.loads(proc.stdout)


class TestGkeJobParams(unittest.TestCase):

    def test_field_validation(self):
        shm = gke_job.gke_common.DEFAULT_SHM_SIZE_GB
        # (field, value, what the config attribute becomes). environment.json
        # answers default to "": a blank is unset, and JSON answers are
        # strings. A blank shm_size_gb takes the default rather than leaving
        # the pod on the runtime's 64 MB.
        accepted = [
            ("namespace", "", gke_job.gke_common.DEFAULT_NAMESPACE),
            ("image", "", None),
            ("tpu_chips", "8", 8),
            ("declared_duration_minutes", "15", 15),
            ("declared_duration_minutes", 0, None),
            ("shm_size_gb", "", shm),
            ("shm_size_gb", "128", 128),
            ("hf_cache_pvc", "  ", None),
            ("hf_cache_pvc", " hf-cache.shared ", "hf-cache.shared"),
            ("compile_cache_max_age_days", 0, 0),
            ("compile_cache_url", " gs://b/xla_cache/ ", "gs://b/xla_cache"),
            ("kube_context", " ctx-a ", "ctx-a"),
            ("job_name_prefix", " ", None),
            ("job_name_prefix", "Agent_Evals", "agent-evals"),
        ]
        for field, value, want in accepted:
            with self.subTest(field=field, value=value):
                driver = gke_job.GkeJobDriver.from_params({field: value})
                self.assertEqual(getattr(driver.config, field), want)
        rejected = [
            ("tpu_chips", "four"),
            # A pod that requests no chips silently computes on the CPU.
            ("tpu_chips", 0),
            ("priority", "high"),
            ("declared_duration_minutes", -1),
            ("shm_size_gb", 0),
            ("hf_cache_pvc", "HF_Cache"),
            ("hf_cache_pvc", "a" * 254),
            ("compile_cache_max_age_days", -1),
            ("compile_cache_url", "/cache"),
            # A bare bucket: a patched build's -patch-<key> would name another.
            ("compile_cache_url", "gs://b/"),
            ("job_name_prefix", "___"),
            ("job_name_prefix",
             "a" * (gke_job.profile_settings.MAX_JOB_NAME_PREFIX_LEN + 1)),
        ]
        for field, value in rejected:
            with self.subTest(field=field, value=value), \
                    self.assertRaises(ComputeConfigError) as ctx:
                gke_job.GkeJobDriver.from_params({field: value})
            self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")


class TestGkeJobName(unittest.TestCase):
    """How `_job_name` turns a requested name into a Job name."""

    @patch.object(jobs, "new_job_id", return_value="job-1")
    def test_names_are_cleaned_and_prefixed_once(self, _):
        plain, agent = make_driver(), make_driver(job_name_prefix="agent")
        for driver, requested, unique, want in (
            (plain, None, True, "job-1"),  # A fresh id needs no suffix.
            (plain, " My_Job.v2 ", False, "my-job-v2"),
            (agent, None, False, "agent-job-1"),
            (agent, "bench", False, "agent-bench"),
            (agent, "agent-bench", False, "agent-bench"),
            (agent, "agent", False, "agent"),
        ):
            self.assertEqual(driver._job_name(requested, unique), want)
        with self.assertRaises(ComputeConfigError) as ctx:
            plain._job_name("___", unique=False)
        self.assertEqual(ctx.exception.error_code, "INVALID_FIELD")
        # A unique suffix survives trimming, and naming the result again
        # (as `submit` does with the name `run` picked) leaves it alone.
        name = agent._job_name("Bench_" + "x" * 100, unique=True)
        self.assertEqual(len(name), gke_job.profile_settings.MAX_JOB_NAME_LEN)
        self.assertRegex(name, r"^agent-bench-x+-[0-9a-f]{6}$")
        self.assertEqual(agent._job_name(name, unique=False), name)


class TestGkeJobSubmit(unittest.TestCase):

    def setUp(self):
        self.script = hello_script(self)
        self.tmp = self.script.parent
        self.enterContext(job_store())

    def submit(self, fake: FakeKubectl, **overrides: Any) -> str:
        driver = make_driver(**overrides)
        with patch.object(subprocess, "run", fake):
            return driver.submit(str(self.script), job_name="run-1")

    def test_a_name_held_outside_the_run_gets_a_suffix(self):
        # Refusing with "cancel it if it is yours" led an agent to cancel
        # another run's job.
        taken = "get configmap,job agent-run-1 "
        fake = FakeKubectl({taken: (0, "job.batch/agent-run-1\n", "")})
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            job_id = self.submit(fake, job_name_prefix="agent")
        self.assertRegex(job_id, r"^agent-run-1-[0-9a-f]{6}$")
        self.assertEqual(fake.applied_manifest()["metadata"]["name"], job_id)
        self.assertIn(f"submitting as {job_id!r}", err.getvalue())
        self.assertIsNone(fake.find("delete"))

    def test_a_name_held_by_this_runs_live_job_is_refused(self):
        jobs.JobRecord.create(job_id="agent-run-1",
                              backend="gke_job",
                              run_id="run-a")
        fake = FakeKubectl({"get configmap,job": (0, "job.batch/x\n", "")})
        driver = make_driver(job_name_prefix="agent")
        with patch.object(subprocess, "run", fake), \
                self.assertRaises(ComputeConfigError) as ctx:
            driver.submit(str(self.script), job_name="run-1", run_id="run-a")
        self.assertEqual(ctx.exception.error_code, "NAME_TAKEN")
        self.assertIn("already this run's job", ctx.exception.message)
        self.assertEqual(len(fake.calls), 1)

    def test_a_name_whose_suffixes_are_all_taken_is_refused(self):
        fake = FakeKubectl({"get configmap,job": (0, "job.batch/x\n", "")})
        with self.assertRaises(ComputeConfigError) as ctx:
            self.submit(fake, job_name_prefix="agent")
        self.assertEqual(ctx.exception.error_code, "NAME_TAKEN")
        self.assertIn("'agent-run-1' (from --name 'run-1')",
                      ctx.exception.message)
        self.assertNotIn("job cancel", ctx.exception.message)
        # The name, then each suffixed candidate; nothing created.
        self.assertEqual(len(fake.calls),
                         1 + gke_job.driver.NAME_SUFFIX_ATTEMPTS)
        self.assertIsNone(fake.find("create"))

    def test_manifest_mounts_the_hf_cache_pvc(self):
        # Without it each pod starts with an empty filesystem and downloads
        # the model again: a 65 GB Qwen took ~50 minutes per pod.
        fake = FakeKubectl()
        self.submit(fake, hf_cache_pvc="hf-cache")
        pod_spec = fake.applied_pod_spec()
        container = pod_spec["containers"][0]
        mounts = {m["name"]: m["mountPath"] for m in container["volumeMounts"]}
        volumes = {v["name"]: v for v in pod_spec["volumes"]}
        self.assertEqual(mounts["hf-cache"], "/hf-cache")
        self.assertEqual(volumes["hf-cache"]["persistentVolumeClaim"],
                         {"claimName": "hf-cache"})
        # Service links put a shared namespace's Services in the pod's env
        # (VLLM_..._PORT=tcp://...), which vLLM misreads as its settings.
        self.assertIs(pod_spec["enableServiceLinks"], False)
        # The hub cache only: HF_HOME would also put the token on it.
        env = {e["name"]: e["value"] for e in container["env"]}
        env.pop("TPU_AGENT_SUITE_SHM_LIMIT_BYTES")
        self.assertEqual(env, {"HF_HUB_CACHE": "/hf-cache"})
        # The compile caches are set in the container, not the pod spec,
        # which would override a cache the image's ENV points at.
        prologue = container["command"][:4]
        cache = "/hf-cache/compile-cache/tpu7x-2x2x1/sha256-abc"
        names = ("VLLM_CACHE_ROOT",
                 "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT",
                 "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT")
        cases = [
            # (the image's ENV, what the workload sees)
            ({}, {
                "VLLM_CACHE_ROOT": f"{cache}/vllm",
                "TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT":
                f"{cache}/torch_tpu",
                "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": f"{cache}/torch_tpu",
            }),
            # An image with its own caches keeps them; torch_tpu's two
            # spellings move together.
            ({
                "VLLM_CACHE_ROOT": "/opt/vllm",
                "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": "/opt/tpu",
            }, {
                "VLLM_CACHE_ROOT": "/opt/vllm",
                "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": "/opt/tpu",
            }),
        ]
        for image_env, seen in cases:
            with self.subTest(image_env=image_env):
                self.assertEqual(env_behind(prologue, image_env, names), seen)
        self.assertRegex(gke_job.manifest.compile_cache_key("gcr.io/p/img:v1"),
                         r"^img-v1-[0-9a-f]{12}$")

    def test_compile_cache_url_is_every_pods_tier3_root(self):
        # Wrapped outside hf_cache_pvc's defaults, which then leave
        # torch_tpu's roots to it; an image's own root still wins.
        url, backup = "gs://b/xla_cache", gke_job.manifest.TIER3_BACKUP_ENV
        fake = FakeKubectl()
        self.submit(fake, compile_cache_url=url, hf_cache_pvc="hf-cache")
        prologue = fake.applied_pod_spec()["containers"][0]["command"][:8]
        names = ("VLLM_CACHE_ROOT", "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT",
                 backup)
        for image_env, tier3, want_backup in (
            ({}, url, "0"),
            ({
                "TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT": "/opt/tpu"
            }, "/opt/tpu", None),
        ):
            with self.subTest(image_env=image_env):
                seen = env_behind(prologue, image_env, names)
                self.assertTrue(
                    seen["VLLM_CACHE_ROOT"].startswith("/hf-cache/"))
                self.assertEqual(
                    seen["TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT"], tier3)
                self.assertEqual(seen.get(backup), want_backup)

    def test_compile_cache_prune_removes_only_stale_folders(self):
        mount = self.tmp / "vol"
        shape = mount / "compile-cache" / "tpu7x-2x2x1"
        # (folder, age in days of its file, of its stamp or None, kept)
        folders = [("old", 30, 30, False), ("old-unstamped", 30, None, False),
                   ("restamped", 30, 1, True),
                   ("new-unstamped", 1, None, True), ("current", 30, 30, True)]
        for name, file_age, stamp_age, _ in folders:
            (shape / name / "vllm").mkdir(parents=True)
            for path, age in ((shape / name / "vllm" / "f", file_age),
                              (shape / name / ".last_used", stamp_age)):
                if age is not None:
                    subprocess.run(["touch", "-d", f"{age} days ago", path],
                                   check=True)
            for path in (shape / name / "vllm", shape / name):
                subprocess.run(["touch", "-d", f"{file_age} days ago", path],
                               check=True)
        # Over 90% full: the caches stay in the pod, so `current` is not
        # restamped, and it is still never removed.
        bin_dir = self.tmp / "bin"
        bin_dir.mkdir()
        (bin_dir /
         "df").write_text("#!/bin/sh\necho head\necho fs 100 95 5 95% /vol\n")
        (bin_dir / "df").chmod(0o755)
        command = gke_job.manifest._compile_cache_command([
            sys.executable, "-c", "import os; print(os.environ['"
            "VLLM_CACHE_ROOT'])"
        ], str(shape / "current"), 14, str(mount))
        # Returns once the background prune closes the pipes it inherited.
        proc = subprocess.run(command,
                              env={"PATH": f"{bin_dir}:{os.environ['PATH']}"},
                              capture_output=True,
                              text=True,
                              check=True)
        self.assertEqual(proc.stdout.strip(),
                         gke_job.manifest.POD_CACHE_ENV["VLLM_CACHE_ROOT"][0])
        self.assertIn("95% full", proc.stderr)
        for name, _, _, kept in folders:
            self.assertEqual((shape / name).is_dir(), kept, name)

    def test_kueue_labels_applied_when_queue_configured(self):
        fake = FakeKubectl()
        self.submit(fake,
                    queue="default",
                    priority="low",
                    declared_duration_minutes=15,
                    kubeconfig="/k",
                    kube_context="ctx-a")
        # Every call names the cluster: without --context, the wrong one.
        for cmd in fake.calls:
            self.assertEqual(cmd[1:5],
                             ["--kubeconfig", "/k", "--context", "ctx-a"])
        manifest = fake.applied_manifest()
        labels = manifest["metadata"]["labels"]
        self.assertEqual(labels[gke_job.gke_common.QUEUE_LABEL], "default")
        self.assertEqual(labels[gke_job.gke_common.PRIORITY_LABEL], "low")

        pod_labels = manifest["spec"]["template"]["metadata"]["labels"]
        self.assertEqual(pod_labels["namespace"], "default")
        self.assertEqual(pod_labels["priority"], "low")
        self.assertEqual(pod_labels["declared-duration-minutes"], "15")
        # Kueue refuses a Job naming a WorkloadPriorityClass that does not
        # exist, so no priority chosen means no priority label at all.
        fake = FakeKubectl()
        self.submit(fake, queue="default", priority="")
        manifest = fake.applied_manifest()
        self.assertNotIn(gke_job.gke_common.PRIORITY_LABEL,
                         manifest["metadata"]["labels"])
        self.assertNotIn("priority",
                         manifest["spec"]["template"]["metadata"]["labels"])

    def test_colocate_takes_the_servers_placement_without_chips(self):
        server = {
            "nodeName": "n1",
            "nodeSelector": {
                "flavor": "tpu"
            },
            "tolerations": [{
                "key": "reservation"
            }]
        }
        pods = json.dumps({"items": [{"spec": server}]})
        fake = FakeKubectl({"get pods": (0, pods, "")})
        self.submit(fake, colocate_with="srv", colocate_requests="cpu=1")
        pod_spec = fake.applied_pod_spec()
        self.assertEqual(pod_spec["nodeSelector"], {
            "flavor": "tpu",
            gke_job.gke_common.HOSTNAME_LABEL: "n1"
        })
        self.assertEqual(pod_spec["tolerations"], server["tolerations"])
        self.assertEqual(pod_spec["containers"][0]["resources"],
                         {"requests": {
                             "cpu": "1"
                         }})
        with self.assertRaises(ComputeConfigError):
            make_driver(colocate_requests="cpu")
        with self.assertRaises(ComputeConfigError):
            make_driver(colocate_with="srv")

    def test_cpu_requests_book_no_tpu_node(self):
        """No chips, and nothing that sends the pod to a TPU node.

        Only the Workload Identity selector, no TPU toleration. Not with
        colocation. The profile's TPU generation is passed on, as probe names
        it, for a check that plans for that TPU without a chip.
        """
        fake = FakeKubectl()
        self.submit(fake,
                    cpu_requests="cpu=2,memory=8Gi",
                    pod_service_account="agent-sa")
        pod_spec = fake.applied_pod_spec()
        self.assertEqual(pod_spec["nodeSelector"],
                         {gke_job.gke_common.WORKLOAD_IDENTITY_LABEL: "true"})
        self.assertNotIn("tolerations", pod_spec)
        self.assertEqual(pod_spec["containers"][0]["resources"],
                         {"requests": {
                             "cpu": "2",
                             "memory": "8Gi"
                         }})
        env = {
            e["name"]: e.get("value")
            for e in fake.applied_container()["env"]
        }
        self.assertEqual(env[gke_job.gke_common.CHIPLESS_TPU_ENV], "tpu7x")
        with self.assertRaises(ComputeConfigError):
            make_driver(cpu_requests="cpu=2",
                        colocate_with="srv",
                        colocate_requests="cpu=1")

    def test_a_cpu_run_that_needs_a_chip_runs_again_on_a_tpu_node(self):
        """It runs again only once, and only when it needs a chip.

        That is when no CPU node took it or it exited NEEDS_TPU_EXIT_CODE; any
        other result is the run's own.
        """

        def ran(code, **details):
            return base.SynchronousExecutionResult(exit_code=code,
                                                   stdout="",
                                                   stderr="",
                                                   details=details)

        cases = [
            (ran(1, never_started=True), "no CPU node took it"),
            (ran(gke_job.gke_common.NEEDS_TPU_EXIT_CODE), "it exited 75"),
            (ran(1), None),
        ]
        for first, why in cases:
            with self.subTest(why=why):
                driver = make_driver(cpu_requests="cpu=2")
                results, held = [first, ran(0)], []

                def once(**_kwargs):
                    held.append(dict(driver.config.cpu_requests))
                    return results.pop(0)

                with patch.object(driver, "_run_once", side_effect=once), \
                        contextlib.redirect_stderr(io.StringIO()):
                    result = driver.run(str(self.script))
                if why is None:
                    self.assertEqual(held, [{"cpu": "2"}])
                    self.assertIs(result, first)
                    continue
                self.assertEqual(held, [{"cpu": "2"}, {}])
                self.assertEqual(result.exit_code, 0)
                self.assertIn(why, result.details["cpu_fallback"])
                self.assertEqual(driver.config.cpu_requests, {"cpu": "2"})

    def test_a_refused_service_keeps_the_job_and_its_record(self):
        """RBAC refusing the Service warns; the running Job stays tracked.

        A 403 goes through the ADC retry and comes back as
        ClusterAccessDenied, not KubectlError. It used to escape submit()
        after the Job was applied, delete the Job's ConfigMap in the cleanup
        and leave no record of the job id.
        """
        forbidden = ('Error from server (Forbidden): services is forbidden: '
                     'User "x" cannot create resource "services"')

        class RefuseService(FakeKubectl):

            def __call__(self, cmd, **kwargs):
                if '"kind": "Service"' in (kwargs.get("input") or ""):
                    self.calls.append(cmd)
                    self.stdins.append(kwargs.get("input"))
                    return subprocess.CompletedProcess(cmd, 1, "", forbidden)
                return super().__call__(cmd, **kwargs)

        fake = RefuseService({"jsonpath": (0, "uid-1", "")})
        driver = make_driver()
        with patch.object(subprocess, "run", fake), \
                patch.object(auth_utils, "adc_token",
                             return_value=("", "no ADC")), \
                patch("sys.stderr", io.StringIO()) as err:
            job_id = driver.submit(str(self.script),
                                   job_name="run-1",
                                   serve=True)
        self.assertEqual(job_id, "run-1")
        self.assertIsNotNone(jobs.JobRecord.load("run-1"))
        self.assertIsNone(fake.find("delete configmap"))
        self.assertIn("no Service for run-1", err.getvalue())
        # A service that a single crash ends is not a service.
        self.assertEqual(fake.applied_pod_spec()["restartPolicy"], "OnFailure")
        self.assertEqual(fake.applied_manifest()["spec"]["backoffLimit"],
                         gke_job.gke_common.SERVE_BACKOFF_LIMIT)


if __name__ == "__main__":
    unittest.main()
