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
"""Runs the stdlib-only serving hill-climb gate tests in CI test discovery."""

import importlib.util
import unittest
from pathlib import Path

_HILLCLIMB_TEST = (Path(__file__).resolve().parents[1] / "scripts" /
                   "optimization-heuristics" / "hillclimb" /
                   "hillclimb_test.py")


def _load_script_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_hillclimb_test = _load_script_module("hillclimb_test", _HILLCLIMB_TEST)


class TestHillclimbMetricsAndHistory(_hillclimb_test.MetricsAndHistoryTest):
    """Wire MetricsAndHistoryTest into package discovery."""


class TestHillclimbDisaggGates(_hillclimb_test.DisaggGatesAndReplayTest):
    """Wire DisaggGatesAndReplayTest into package discovery."""


class TestHillclimbSourceAudit(_hillclimb_test.SourceAuditAndMicrobenchTest):
    """Wire SourceAuditAndMicrobenchTest into package discovery."""


class TestHillclimbRooflinePredictor(_hillclimb_test.RooflineAndPredictorTest):
    """Wire RooflineAndPredictorTest into package discovery."""


class TestHillclimbCli(_hillclimb_test.HillclimbCliAllSubcommandsTest):
    """Wire HillclimbCliAllSubcommandsTest into package discovery."""


if __name__ == "__main__":
    unittest.main()
