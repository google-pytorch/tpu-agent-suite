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
"""Throughput improvement over the baseline: the task's reward."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure  # noqa: E402  (one shared, cached measurement)
from rewardkit import criterion  # noqa: E402


@criterion(
    description="speedup - 1 over the baseline, for a valid, gated run; else 0"
)
def throughput_improvement(workspace: Path) -> dict:
    r = measure.evaluate(workspace)
    return {
        "score":
        measure.improvement_score(r),
        "reasoning":
        f"speedup={r.get('speedup')} "
        f"baseline_p50={r.get('baseline_step_time_p50_s')} "
        f"candidate_p50={r.get('candidate_step_time_p50_s')}"
    }
