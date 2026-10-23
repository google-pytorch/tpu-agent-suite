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
"""The compute answers a package needs, read from an agent's environment."""

from __future__ import annotations

import json
import os
from typing import Any


def collect_task_environment_answers(
    extra_env: dict[str, str] | None = None, ) -> dict[str, Any]:
    """Collect environment answers from agent extra_env or task environment variables."""
    env_lookup = dict(os.environ)
    if extra_env:
        env_lookup.update(extra_env)

    backend = env_lookup.get("COMPUTE_BACKEND", "gke_job")

    answers: dict[str, Any] = {
        "backend": backend,
    }

    if backend == "local":
        pass

    elif backend == "gce_tpu_vm":
        answers.update({
            "tpu_name": env_lookup.get("GCE_TPU_NAME", ""),
            "zone": env_lookup.get("GCE_LOCATION", ""),
            "project": env_lookup.get("GCE_PROJECT", ""),
        })

    elif backend == "gke_pod":
        answers.update({
            "pod": env_lookup.get("GKE_POD", ""),
            "namespace": env_lookup.get("GKE_NAMESPACE", "default"),
            "container": env_lookup.get("GKE_CONTAINER", ""),
        })

    elif backend == "gke_job":
        duration = env_lookup.get("GKE_DECLARED_DURATION_MINUTES", "")
        answers.update({
            "image":
            env_lookup.get("TASK_DOCKER", ""),
            "tpu_accelerator":
            env_lookup.get("GKE_TPU_ACCELERATOR", ""),
            "tpu_topology":
            env_lookup.get("GKE_TPU_TOPOLOGY", ""),
            "queue":
            env_lookup.get("GKE_QUEUE", ""),
            "priority":
            env_lookup.get("GKE_PRIORITY", ""),
            "namespace":
            env_lookup.get("GKE_NAMESPACE", ""),
            "declared_duration_minutes":
            int(duration) if duration.isdigit() else 0,
            "kubeconfig":
            env_lookup.get("KUBECONFIG", ""),
            # Staging through GCS with the pod's Workload Identity token, for
            # inputs and outputs too big for a ConfigMap or the pod's log.
            "staging_bucket":
            env_lookup.get("GKE_STAGING_BUCKET", ""),
            "pod_service_account":
            env_lookup.get("GKE_POD_SERVICE_ACCOUNT", ""),
            "compile_cache_url":
            env_lookup.get("GKE_COMPILE_CACHE_URL", ""),
        })

    raw_extra_answers = env_lookup.get("COMPUTE_ANSWERS_JSON", "").strip()
    if raw_extra_answers:
        extra_answers = json.loads(raw_extra_answers)
        if not isinstance(extra_answers, dict):
            raise ValueError(
                "COMPUTE_ANSWERS_JSON must decode to a JSON object")
        answers.update(extra_answers)

    return answers
