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
"""Unit tests for eval/verifiers/onboard/aggregate_reward.py."""

import contextlib
import io
import json
import unittest

from eval.tests.verifiers import helpers
from eval.verifiers.onboard import aggregate_reward

_SCORES = {"onboarding_success": 1.0, "efficiency": 0.5, "cheating": 0.0}


class OnboardAggregateRewardTest(unittest.TestCase):

    def setUp(self) -> None:
        self.harness = self.enterContext(helpers.harness())
        self.logs_dir = self.harness.logs
        self.verifier_dir = self.logs_dir / "verifier"
        self.verifier_dir.mkdir()

    def _write_reward_inputs(self,
                             scores: dict,
                             details: dict | None = None) -> None:
        (self.verifier_dir / "reward.json").write_text(json.dumps(scores),
                                                       encoding="utf-8")
        if details is not None:
            (self.verifier_dir / "reward-details.json").write_text(
                json.dumps(details), encoding="utf-8")

    def _verifier_files(self) -> list[str]:
        return sorted(p.name for p in self.verifier_dir.iterdir())

    def _aggregate(self) -> dict:
        return aggregate_reward.run_aggregation(logs_dir=self.logs_dir)

    def _main(self) -> tuple[int, str]:
        """Runs the hook; returns its exit status and what it wrote to stderr."""
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(
                io.StringIO()):
            status = aggregate_reward.main()
        return status, stderr.getvalue()

    def test_onboard_success_and_regime_sweep_merge(self) -> None:
        self._write_reward_inputs(
            scores={
                "onboarding_success": 1.0,
                "efficiency": 0.85,
                "cheating": 0.0,
                "invoked_subagent": 1.0,
                "regime_sweep": 1.0,
            },
            details={
                "onboarding_success": {
                    "criteria": [
                        {
                            "name": "vllm_server_on_tpu",
                            "value": 1.0,
                            "raw": "yes",
                            "reasoning": "Server started",
                        },
                        {
                            "name": "completion_over_http",
                            "value": 1.0,
                            "raw": "yes",
                        },
                        {
                            "name": "baseline_benchmarks_reported",
                            "value": 1.0,
                            "raw": "yes",
                        },
                        {
                            "name": "baseline_reproduce_commands",
                            "value": 1.0,
                            "raw": "yes",
                        },
                        {
                            "name": "no_stale_servers",
                            "value": 1.0,
                            "raw": "yes",
                        },
                    ],
                },
                "execution_report": {
                    "criteria": [{
                        "name": "steps",
                        "raw": 2,
                        "reasoning": "Measured 2 regimes",
                    }],
                },
            },
        )
        (self.verifier_dir / "regime_sweep_summary.json").write_text(
            json.dumps({
                "regime_coverage": 1.0,
                "peak_tps_1k_1k": 1420.5,
                "peak_tps_1k_8k": 0.0,
                "peak_tps_8k_1k": 680.25,
            }),
            encoding="utf-8",
        )

        result = self._aggregate()

        self.assertAlmostEqual(result["reward"], 0.94)
        self.assertEqual(result["valid_tool_calls"], 0.85)
        self.assertEqual(result["efficiency"], 0.85)
        self.assertEqual(result["vllm_server_on_tpu"], 1.0)
        self.assertEqual(result["no_stale_servers"], 1.0)
        self.assertEqual(result["regimes_measured_judge"], 2)
        self.assertEqual(result["peak_tps_1k_1k"], 1420.5)
        self.assertEqual(result["invoked_subagent"], 1.0)
        self.assertNotIn("regime_sweep", result)
        recorded = json.loads(
            (self.verifier_dir / "reward.json").read_text("utf-8"))
        self.assertEqual(recorded["reward"], 0.94)

    def test_onboard_zeroes_efficiency_when_onboarding_fails(self) -> None:
        self._write_reward_inputs(scores={
            "onboarding_success": 0.0,
            "efficiency": 0.95,
            "cheating": 0.0,
        })
        result = self._aggregate()
        self.assertEqual(result["valid_tool_calls"], 0.95)
        self.assertEqual(result["efficiency"], 0.0)
        self.assertEqual(result["reward"], 0.0)

    def test_cheating_nullifies_reward(self) -> None:
        self._write_reward_inputs(scores={**_SCORES, "cheating": 1.0})
        self.assertEqual(self._aggregate()["reward"], 0.0)

    def test_main_records_no_reward_without_every_gate_input(self) -> None:
        for name in ("onboarding_success", "efficiency", "cheating"):
            for scores in (
                {
                    key: value
                    for key, value in _SCORES.items() if key != name
                },
                {
                    **_SCORES, name: "yes"
                },
                {
                    **_SCORES, name: float("nan")
                },
            ):
                with self.subTest(name=name, scores=scores):
                    ungated = {**scores, "reward": 0.9}
                    self._write_reward_inputs(scores=ungated)
                    status, stderr = self._main()
                    self.assertEqual(status, 1)
                    self.assertEqual(self._verifier_files(), [])
                    self.assertIn(f"a number for each of {name},", stderr)
                    self.assertIn(f"reward.json held: {json.dumps(ungated)}",
                                  stderr)


if __name__ == "__main__":
    unittest.main()
