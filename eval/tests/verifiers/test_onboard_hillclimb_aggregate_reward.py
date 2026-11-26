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
"""Unit tests for eval/verifiers/onboard_hillclimb/aggregate_reward.py."""

import contextlib
import io
import json
import math
import unittest

from eval.tests.verifiers import helpers
from eval.verifiers.onboard_hillclimb import aggregate_reward

_SCORES = {
    "onboarding_success": 1.0,
    "speedup": 1.25,
    "efficiency": 0.8,
    "cheating": 0.0,
    "invoked_subagent": 1.0,
}
# The reward of _SCORES against the default target_speedup of 1.0.
_REWARD = round(0.6 + 0.3 * (math.log(1.25) / math.log(2.0)) + 0.1 * 0.8, 4)


class HillclimbAggregateRewardTest(unittest.TestCase):

    def setUp(self) -> None:
        self.harness = self.enterContext(helpers.harness())
        self.logs_dir = self.harness.logs
        self.tests_dir = self.harness.tests
        self.verifier_dir = self.logs_dir / "verifier"
        self.verifier_dir.mkdir()

    def _write_target_speedup(self, target: float) -> None:
        self.harness.write_params({"speedup": {"target_speedup": target}})

    def _write_scores(self, scores: dict) -> None:
        (self.verifier_dir / "reward.json").write_text(json.dumps(scores),
                                                       encoding="utf-8")

    def _verifier_files(self) -> list[str]:
        return sorted(p.name for p in self.verifier_dir.iterdir())

    def _aggregate(self) -> dict:
        return aggregate_reward.run_aggregation(logs_dir=self.logs_dir,
                                                tests_dir=self.tests_dir)

    def _recorded_reward(self) -> float:
        """The reward in reward.json, the only reward file the hook writes."""
        self.assertEqual(self._verifier_files(), ["reward.json"])
        return json.loads(
            (self.verifier_dir / "reward.json").read_text("utf-8"))["reward"]

    def _main(self) -> tuple[int, str]:
        """Runs the hook; returns its exit status and what it wrote to stderr."""
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr), contextlib.redirect_stdout(
                io.StringIO()):
            status = aggregate_reward.main()
        return status, stderr.getvalue()

    def test_hillclimb_gated_and_log_scaled(self) -> None:
        self._write_scores(_SCORES)
        result = self._aggregate()
        self.assertEqual(result["hillclimb_success"], 1.0)
        self.assertEqual(result["speedup_score"],
                         round(math.log(1.25) / math.log(2.0), 4))
        self.assertEqual(result["reward"], _REWARD)
        self.assertEqual(self._recorded_reward(), _REWARD)

    def test_hillclimb_below_one_percent_earns_only_onboarding(self) -> None:
        self._write_target_speedup(1.0)
        for scores in ({
                **_SCORES, "speedup": 1.005
        }, {
                key: value
                for key, value in _SCORES.items() if key != "speedup"
        }):
            with self.subTest(scores=scores):
                self._write_scores(scores)
                result = self._aggregate()
                self.assertEqual(result["hillclimb_success"], 0.0)
                self.assertEqual(result["speedup_score"], 0.0)
                self.assertEqual(result["efficiency"], 0.0)
                self.assertEqual(result["reward"], 0.3)

    def test_cheating_nullifies_reward(self) -> None:
        self._write_scores({**_SCORES, "cheating": 1.0})
        self.assertEqual(self._aggregate()["reward"], 0.0)

    def test_target_speedup_without_headroom_is_rejected(self) -> None:
        self._write_scores(_SCORES)
        for target in (0.5, 0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(target=target):
                self._write_target_speedup(target)
                with self.assertRaises(ValueError):
                    self._aggregate()

    def test_target_is_read_only_from_the_tests_dir(self) -> None:
        workspace_tests = self.harness.workspace / "tests"
        workspace_tests.mkdir()
        (workspace_tests / "verifier_params.json").write_text(json.dumps(
            {"speedup": {
                "target_speedup": 0.51
            }}),
                                                              encoding="utf-8")
        self._write_scores(_SCORES)
        with contextlib.chdir(self.harness.workspace):
            self.assertEqual(self._aggregate()["target_speedup"], 1.0)

    def test_target_outside_the_speedup_table_is_ignored(self) -> None:
        self.harness.write_params({"target_speedup": 4.0})
        self._write_scores(_SCORES)
        self.assertEqual(self._aggregate()["target_speedup"], 1.0)

    def test_main_reads_the_harness_directories(self) -> None:
        self._write_target_speedup(4.0)
        self._write_scores(_SCORES)
        status, stderr = self._main()
        self.assertEqual((status, stderr), (0, ""))
        scores = json.loads(
            (self.verifier_dir / "reward.json").read_text("utf-8"))
        self.assertEqual(scores["target_speedup"], 4.0)

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
                    self._write_scores(ungated)
                    status, stderr = self._main()
                    self.assertEqual(status, 1)
                    self.assertEqual(self._verifier_files(), [])
                    self.assertIn(f"a number for each of {name},", stderr)
                    self.assertIn(f"reward.json held: {json.dumps(ungated)}",
                                  stderr)

    def test_post_hillclimb_xprof_recorded_in_reward_details(self) -> None:
        details_path = self.verifier_dir / "reward-details.json"
        details_path.write_text(json.dumps({}), encoding="utf-8")
        self._write_scores(_SCORES)

        # Without XProf artifacts -> passed=False, reward unchanged.
        result = aggregate_reward.run_aggregation(
            logs_dir=self.logs_dir,
            tests_dir=self.tests_dir,
            workspace_dir=self.harness.workspace,
        )
        self.assertEqual(result["reward"], _REWARD)
        details = json.loads(details_path.read_text(encoding="utf-8"))
        self.assertEqual(
            details["post_hillclimb_xprof"],
            {
                "xplane_trace_count": 0,
                "gaps_report_count": 0,
                "passed": False,
            },
        )

        # Baseline/onboarding/tpu-diagnostics traces and tiny/empty files do NOT pass.
        for ignored_rel in (
                "onboarding/xprof/base.xplane.pb",
                "tpu-diagnostics/base.xplane.pb",
                "hillclimb/xprof/baseline.xplane.pb",
                "hillclimb/xprof/empty.xplane.pb",
        ):
            p = self.logs_dir / "artifacts" / ignored_rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(
                b"" if ignored_rel.endswith("empty.xplane.pb") else b"xplane")
        (self.harness.workspace / "xprof_bottlenecks_and_gaps.md").write_text(
            "too short", encoding="utf-8")
        self._write_scores(_SCORES)
        aggregate_reward.run_aggregation(
            logs_dir=self.logs_dir,
            tests_dir=self.tests_dir,
            workspace_dir=self.harness.workspace,
        )
        details = json.loads(details_path.read_text(encoding="utf-8"))
        self.assertEqual(
            details["post_hillclimb_xprof"],
            {
                "xplane_trace_count": 0,
                "gaps_report_count": 0,
                "passed": False,
            },
        )

        # With non-empty post-hill-climb .xplane.pb under hillclimb/ and >=100B
        # gaps report + speedup_summary.json -> both recorded in reward-details.json.
        xprof_dir = self.logs_dir / "artifacts" / "hillclimb" / "xprof"
        xprof_dir.mkdir(parents=True, exist_ok=True)
        (xprof_dir / "final.xplane.pb").write_bytes(b"xplane")
        (self.harness.workspace / "xprof_bottlenecks_and_gaps.md").write_text(
            "x" * 150, encoding="utf-8")
        summary_payload = {
            "primary_regime": "8k/1k",
            "per_chip": True,
            "baseline_num_chips": 4,
            "optimized_num_chips": 4,
            "baseline_peak_tps_per_chip": 300.0,
            "optimized_peak_tps_per_chip": 450.0,
            "verified_speedup": 1.5,
            "verified_manifest_path": "/logs/artifacts/benchmark_results.json",
        }
        (self.verifier_dir / "speedup_summary.json").write_text(
            json.dumps(summary_payload), encoding="utf-8")
        self._write_scores(_SCORES)
        result = aggregate_reward.run_aggregation(
            logs_dir=self.logs_dir,
            tests_dir=self.tests_dir,
            workspace_dir=self.harness.workspace,
        )
        self.assertEqual(result["reward"], _REWARD)
        details = json.loads(details_path.read_text(encoding="utf-8"))
        self.assertEqual(
            details["post_hillclimb_xprof"],
            {
                "xplane_trace_count": 1,
                "gaps_report_count": 1,
                "passed": True,
            },
        )
        self.assertEqual(details["speedup_summary"], summary_payload)

        # Malformed reward-details.json is preserved without overwriting.
        raw_corrupt = "{broken json"
        details_path.write_text(raw_corrupt, encoding="utf-8")
        self._write_scores(_SCORES)
        aggregate_reward.run_aggregation(
            logs_dir=self.logs_dir,
            tests_dir=self.tests_dir,
            workspace_dir=self.harness.workspace,
        )
        self.assertEqual(details_path.read_text(encoding="utf-8"), raw_corrupt)


if __name__ == "__main__":
    unittest.main()
