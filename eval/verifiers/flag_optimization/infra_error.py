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
"""1 when a zero is the infrastructure's fault (no baseline, queue timeout), not the agent's."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure  # noqa: E402  (one shared, cached measurement)
from rewardkit import criterion  # noqa: E402


@criterion(description="the measurement failed for infrastructure reasons")
def infra_error(workspace: Path) -> dict:
    r = measure.evaluate(workspace)
    return {
        "score": bool(r.get("infra_error")),
        "reasoning": r.get("error", "") if r.get("infra_error") else ""
    }
