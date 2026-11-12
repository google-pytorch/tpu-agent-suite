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
"""Shared fixtures for the study store tests.

Every test points TPU_AGENT_SUITE_HOME at a temp dir, so nothing here touches a
real ~/.tpu-agent-suite.
"""

from __future__ import annotations

import os
import pathlib
import tempfile
import unittest
from unittest import mock

from framework import study

Path = pathlib.Path


class StudyStoreTestCase(unittest.TestCase):
    """Base that isolates the central store into a temp directory."""

    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.home = Path(tmp.name)
        env = mock.patch.dict(os.environ, {"TPU_AGENT_SUITE_HOME": tmp.name})
        env.start()
        self.addCleanup(env.stop)

    @staticmethod
    def protocol(**overrides) -> dict:
        """A minimal protocol that satisfies the schema.

        Built here rather than pasted into forty tests so that adding a
        required field breaks one place, not forty, and so no test can
        accidentally assert the behaviour of a protocol that would be
        rejected in practice.
        """
        base = dict(primary_metric=dict(name="tokens_per_s",
                                        unit="tok/s",
                                        direction="maximize"),
                    measurement_cell=dict(topology="v6e-2x2x1"),
                    replicate_count=3,
                    noise_floor_pct=0.4,
                    stopping_rule=dict(max_rounds=10),
                    admissibility_gates=[],
                    mode="approve")
        base.update(overrides)
        return base

    def start(self, study_id: str, **overrides):
        """Open a study with a valid protocol and identity."""
        return study.start_study(study_id,
                                 protocol=self.protocol(**overrides),
                                 package="tpu-agent-suite",
                                 target="test-model",
                                 owner="tester")

    def start_with_arms(self, study_id: str, arms: list[dict]) -> None:
        """Open a study and write `arms` straight into its state."""
        self.start(study_id)
        doc = study.load_study(study_id)
        doc["arms"] = arms
        study.save_study(doc)


def open_arm(arm_id: str, claim: str, **extra) -> dict:
    """An untested arm, as a round or a claim test seeds it."""
    return {
        "arm": arm_id,
        "kind": "kernel-block",
        "status": "open",
        "claim": claim,
        **extra
    }


class RoundRecordTestCase(StudyStoreTestCase):
    """Shared fixtures for the round-record verbs."""

    GATES = [
        dict(name="ttft",
             metric="ttft_p50_ms",
             comparator="lt",
             bound=5000,
             same_execution=True),
        dict(name="accuracy",
             metric="gsm8k",
             comparator="gte",
             bound=0.94,
             same_execution=True),
    ]

    @staticmethod
    def baseline(**overrides) -> dict:
        base = dict(execution_id="j-base",
                    metric=dict(name="tokens_per_s",
                                value=100.0,
                                unit="tok/s",
                                replicates=3,
                                provenance="measured"),
                    profile="baseline/xplane.pb",
                    fingerprint=dict(commit_sha="abc123",
                                     config_digest="sha256:deadbeef",
                                     cell="v6e-2x2x1/8192-1024/c256"))
        metric_overrides = overrides.pop("metric", None)
        base.update(overrides)
        if metric_overrides is not None:
            base["metric"] = {**base["metric"], **metric_overrides}
        return base

    def start_based(self, study_id: str, **overrides):
        """Open a study that already has a baseline, so arms may report a delta.

        Most round tests need one; a delta without a baseline is a
        comparison with nothing on the other side.
        """
        doc = self.start(study_id, **overrides)
        # The baseline must itself meet the pre-registered replicate count,
        # so derive it rather than hardcoding one that a stricter protocol
        # would reject.
        replicates = doc["protocol"]["replicate_count"]
        study.set_baseline(study_id,
                           self.baseline(metric={"replicates": replicates}))
        return doc

    @staticmethod
    def metric(**overrides) -> dict:
        """An arm's primary-metric report; a 5% measured delta by default."""
        return dict(name="tokens_per_s",
                    value=100.0,
                    replicates=3,
                    delta_pct=5.0,
                    provenance="measured") | overrides

    @staticmethod
    def gate(name: str, execution_id: str = "j-1", **extra) -> dict:
        """A passing gate report measured on `execution_id`."""
        return dict(name=name, passed=True, execution_id=execution_id) | extra

    @classmethod
    def arm(cls, **overrides) -> dict:
        result = dict(verdict="tie",
                      execution_id="j-1",
                      metric=cls.metric(),
                      gates=[
                          cls.gate("ttft", value=4500.0),
                          cls.gate("accuracy", value=0.95)
                      ])
        base = open_arm("a1", "fuse the epilogue", status="decided")
        base["result"] = result
        result_overrides = overrides.pop("result", None)
        base.update(overrides)
        if result_overrides is not None:
            base["result"] = {**base["result"], **result_overrides}
        return base
