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
"""Unit tests for installer_staging module."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from eval.agents.installer_staging import collect_task_environment_answers
from eval.utils import find_repo_root


class TestInstallerStaging(unittest.TestCase):

    def test_find_repo_root(self):
        root = find_repo_root()
        self.assertTrue(
            (root / ".git").exists()
            or (root / "package" / "package.release.json").is_file())

    def test_collect_task_environment_answers(self):
        env_vars = {
            "TASK_DOCKER": "us-central1-docker.pkg.dev/proj/img@sha256:abc",
            "GKE_TPU_ACCELERATOR": "tpu7x",
            "GKE_TPU_TOPOLOGY": "2x2x1",
            "GKE_QUEUE": "test-queue",
            "GKE_PRIORITY": "medium",
            "GKE_NAMESPACE": "test-ns",
            "GKE_DECLARED_DURATION_MINUTES": "120",
            "GKE_STAGING_BUCKET": "gs://eval-staging",
            "GKE_POD_SERVICE_ACCOUNT": "tpu-staging-pod",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            answers = collect_task_environment_answers()
            self.assertEqual(answers["backend"], "gke_job")
            self.assertEqual(answers["image"], env_vars["TASK_DOCKER"])
            self.assertEqual(answers["tpu_accelerator"], "tpu7x")
            self.assertEqual(answers["tpu_topology"], "2x2x1")
            self.assertEqual(answers["queue"], "test-queue")
            self.assertEqual(answers["priority"], "medium")
            self.assertEqual(answers["namespace"], "test-ns")
            self.assertEqual(answers["declared_duration_minutes"], 120)
            self.assertEqual(answers["staging_bucket"], "gs://eval-staging")
            self.assertEqual(answers["pod_service_account"], "tpu-staging-pod")

    def test_collect_task_environment_answers_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            answers = collect_task_environment_answers()
            self.assertEqual(answers["backend"], "gke_job")
            self.assertEqual(answers["image"], "")
            self.assertEqual(answers["declared_duration_minutes"], 0)

    def test_collect_task_environment_answers_local(self):
        extra = {"COMPUTE_BACKEND": "local"}
        answers = collect_task_environment_answers(extra_env=extra)
        self.assertEqual(answers, {"backend": "local"})

    def test_collect_task_environment_answers_gce_tpu_vm(self):
        extra = {
            "COMPUTE_BACKEND": "gce_tpu_vm",
            "GCE_TPU_NAME": "test-tpu-vm",
            "GCE_LOCATION": "us-central2-b",
            "GCE_PROJECT": "test-project",
        }
        answers = collect_task_environment_answers(extra_env=extra)
        self.assertEqual(answers["backend"], "gce_tpu_vm")
        self.assertEqual(answers["tpu_name"], "test-tpu-vm")
        self.assertEqual(answers["zone"], "us-central2-b")
        self.assertEqual(answers["project"], "test-project")

    def test_collect_task_environment_answers_gke_pod(self):
        extra = {
            "COMPUTE_BACKEND": "gke_pod",
            "GKE_POD": "my-tpu-pod",
            "GKE_NAMESPACE": "custom-ns",
            "GKE_CONTAINER": "main-worker",
        }
        answers = collect_task_environment_answers(extra_env=extra)
        self.assertEqual(answers["backend"], "gke_pod")
        self.assertEqual(answers["pod"], "my-tpu-pod")
        self.assertEqual(answers["namespace"], "custom-ns")
        self.assertEqual(answers["container"], "main-worker")

    def test_collect_task_environment_answers_external_driver(self):

        class FakeExternalDriver:

            @classmethod
            def collect_environment_answers(cls, env_lookup):
                return {"custom_target": env_lookup.get("EXT_TARGET", "dflt")}

        with patch("framework.compute.drivers.get_driver",
                   return_value=FakeExternalDriver):
            answers = collect_task_environment_answers(
                extra_env={
                    "COMPUTE_BACKEND": "ext_custom",
                    "EXT_TARGET": "cluster-a",
                })
        self.assertEqual(answers, {
            "backend": "ext_custom",
            "custom_target": "cluster-a",
        })


if __name__ == "__main__":
    unittest.main()
