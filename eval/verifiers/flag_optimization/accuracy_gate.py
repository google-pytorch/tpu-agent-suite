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
"""Per-step loss and gradient norm match the baseline's (the math is unchanged)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure  # noqa: E402  (one shared, cached measurement)
from rewardkit import criterion  # noqa: E402


@criterion(
    description=
    "first gate_steps steps match the baseline's loss and gradient norm")
def accuracy_gate(workspace: Path) -> dict:
    r = measure.evaluate(workspace)
    gate = r.get("gate") or {}
    return {
        "score":
        bool(r.get("accuracy_gate")),
        "reasoning":
        "; ".join(gate.get("reasons", [])) or ("pass" if gate else "no run")
    }
