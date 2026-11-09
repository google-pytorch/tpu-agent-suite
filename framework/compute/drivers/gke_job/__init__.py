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
"""Run a TPU workload on GKE as a Kubernetes Job.

The script is mounted from a ConfigMap into the unmodified image, so nothing
is built or installed. Use `gke_pod` to exec into a pod that is already
running.

A slice that fits on one node is one pod. A larger one (e.g. tpu7x `2x2x2`)
is an Indexed Job with one pod per node behind a headless Service. Each pod
gets `TPU_WORKER_ID` (0 is the leader), `TPU_NUM_HOSTS` and `TPU_HEAD_HOST`;
`logs`, `wait` and `port-forward` read the leader.

Needs `kubectl` and a kubeconfig for the cluster. Dispatch never calls
gcloud; preflight and the Cloud Logging fallback do.

Modules:
- `gke_common`: shared constants (labels, mounts, timeouts) and helpers.
- `profile_settings`: `GkeJobConfig`, validated from the profile or flags.
- `manifest`: builds the Job (node selector, chips, caches) and the GCS
  token Secret its pods use.
- `preflight`: checks the cluster, node pool, image and PVC before submit.
- `status`: reads a Job's live status and waits for it to finish.
- `kueue_admission`: explains why Kueue hasn't admitted a Job.
- `pod_diagnostics`: explains a failed pod (OOM, eviction, lost node) or a
  stuck image pull.
- `log_retrieval`: logs from the pod, a saved copy, GCS or Cloud Logging.
- `file_transfer`: sends input files to the pod and collects its outputs.
- `records`: writes a Job's final status to its record; deletes Jobs.
- `driver`: `GkeJobDriver`, which runs and submits Jobs using the
  modules above.

Cluster calls go through `drivers.kubectl.Kubectl`, which `gke_pod` uses
too. Preflight only builds its commands with it and runs them with
`base.probe_run`, so a failed check is reported instead of raised.
"""

from ..kubectl import KubectlError
from .driver import GkeJobDriver
from .profile_settings import GkeJobConfig

__all__ = ["GkeJobConfig", "GkeJobDriver", "KubectlError"]
