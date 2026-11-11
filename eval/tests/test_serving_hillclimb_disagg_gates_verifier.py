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
"""End-to-end tests for serving-hillclimb-disagg-gates fixtures and verifier.

The valid answer and trajectory are built by running hillclimb_cli.py on the
task's fixtures: each CLI call becomes one shell tool call in an ATIF
trajectory whose tool result is the CLI's stdout, as Harbor records it.
"""

import copy
import json
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
TASK_DIR = (REPO_ROOT / "eval" / "datasets" / "close-gaps-gates" /
            "serving-hillclimb-disagg-gates")
WORK_DIR = TASK_DIR / "environment" / "work"
VERIFIER = TASK_DIR / "tests" / "test.sh"
CLI = (REPO_ROOT / "package" / "scripts" / "optimization-heuristics" /
       "hillclimb" / "hillclimb_cli.py")
ARMS = ("arm_prefill_drafter", "arm_sym_dcp8", "arm_cap20_mtp3",
        "arm_asym_dcp")


def _run_cli_text(calls: list, *argv: str) -> str:
    """Runs the CLI and records the call for the trajectory; returns stdout."""
    proc = subprocess.run(
        [sys.executable, str(CLI), *argv],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        raise AssertionError(
            f"CLI failed ({proc.returncode}): {proc.stderr}\n{proc.stdout}")
    calls.append(("python3 hillclimb_cli.py " + " ".join(argv), proc.stdout))
    return proc.stdout


def _run_cli(calls: list, *args: str) -> dict:
    """Runs the CLI with `--json` and records the call for the trajectory."""
    return json.loads(_run_cli_text(calls, *args, "--json"))


def _trajectory(calls: list) -> dict:
    """An ATIF trajectory with one shell tool call per `(command, stdout)`."""
    steps = []
    for i, (command, stdout) in enumerate(calls):
        call_id = f"call_{i}"
        steps.append({
            "source":
            "agent",
            "tool_calls": [{
                "tool_call_id": call_id,
                "function_name": "Bash",
                "arguments": {
                    "command": command
                },
            }],
            "observation": {
                "results": [{
                    "source_call_id": call_id,
                    "content": stdout
                }]
            },
        })
    return {"schema_version": "ATIF-v1.7", "steps": steps}


def _build_answer_from_cli() -> tuple[dict, list]:
    calls: list = []
    metrics = str(WORK_DIR / "metrics" / "server_metrics_export.csv")
    legs_out = _run_cli(calls, "legs", "--server-metrics", metrics)
    cand_spec = json.loads(
        (WORK_DIR / "candidates.json").read_text(encoding="utf-8"))
    arm_decisions: dict[str, dict[str, str]] = {}
    selected = ""

    with tempfile.TemporaryDirectory() as tmp:
        base_path = pathlib.Path(tmp) / "base.json"
        base_path.write_text(json.dumps(cand_spec["baseline_knobs"]),
                             encoding="utf-8")
        for cand in cand_spec["candidates"]:
            arm_id = cand["id"]
            cfg_path = pathlib.Path(tmp) / f"{arm_id}.json"
            cfg_path.write_text(json.dumps(cand["knobs"]), encoding="utf-8")
            pre = _run_cli(
                calls,
                "preflight",
                "--recipe",
                cand_spec["recipe"],
                "--config",
                str(cfg_path),
                "--defaults",
                str(base_path),
                "--repo",
                str(WORK_DIR / "repo"),
                "--server-metrics",
                metrics,
                "--no-fail-on-error",
            )
            findings = pre["findings"]
            checks = {f["check"] for f in findings}
            messages = "; ".join(f["message"] for f in findings)
            if "decode_tp_kernel" in checks:
                arm_decisions[arm_id] = {
                    "decision": "needs_kernel_extension",
                    "reason": messages,
                }
            elif pre["has_errors"]:
                arm_decisions[arm_id] = {
                    "decision": "reject",
                    "reason": messages,
                }
            else:
                arm_decisions[arm_id] = {
                    "decision":
                    "launch",
                    "reason":
                    ("Passes all preflight gates: preserves prefill DCP=16 "
                     "HBM KV cache and lowers decode DCP to 8 inside "
                     "ROW_BUCKETS=(16, 32, 64)."),
                }
                selected = arm_id

    ev = _run_cli(
        calls,
        "early-verdict",
        "--baseline-log",
        str(WORK_DIR / "telemetry" / "baseline_aiperf.txt"),
        "--arm-log",
        str(WORK_DIR / "telemetry" / "inflight_arm_aiperf.txt"),
    )
    answer = {
        "binding_leg":
        "decode" if legs_out["kind"] == "decode_limited" else legs_out["kind"],
        "arm_decisions":
        arm_decisions,
        "selected_arm":
        selected,
        "inflight_action":
        ev["status"],
        "inflight_reason":
        ev["reason"],
    }
    return answer, calls


class ServingHillclimbDisaggGatesVerifierTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.valid_answer, cls.cli_calls = _build_answer_from_cli()
        cls.valid_trajectory = _trajectory(cls.cli_calls)

    def _score(self,
               answer: dict | None,
               trajectory: dict | None = None,
               raw_trajectory: str | None = None) -> tuple[float, dict]:
        """Scores `answer` with `trajectory` (default: the CLI-derived one).

        `raw_trajectory` is written verbatim instead, for malformed files.
        """
        if trajectory is None:
            trajectory = self.valid_trajectory
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            verifier_dir = root / "verifier"
            verifier_dir.mkdir()
            agent_dir = root / "agent"
            agent_dir.mkdir()
            if raw_trajectory is not None:
                (agent_dir / "trajectory.json").write_text(raw_trajectory,
                                                           encoding="utf-8")
            elif trajectory:
                (agent_dir / "trajectory.json").write_text(
                    json.dumps(trajectory), encoding="utf-8")
            ans_path = root / "answer.json"
            if answer is not None:
                ans_path.write_text(json.dumps(answer), encoding="utf-8")
            done = subprocess.run(
                ["bash", str(VERIFIER)],
                env={
                    "PATH": "/usr/bin:/bin:/usr/local/bin",
                    "LOGS_DIR": str(verifier_dir),
                    "ANSWER_JSON": str(ans_path),
                    "AGENT_LOGS_DIR": str(agent_dir),
                },
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            reward_txt = float(
                (verifier_dir /
                 "reward.txt").read_text(encoding="utf-8").strip())
            reward_json = json.loads(
                (verifier_dir /
                 "reward.json").read_text(encoding="utf-8"))["reward"]
            details = json.loads(
                (verifier_dir /
                 "reward-details.json").read_text(encoding="utf-8"))
            self.assertEqual(reward_txt, reward_json)
            return reward_txt, details

    def test_cli_derived_answer_scores_full_reward(self) -> None:
        score, details = self._score(self.valid_answer)
        self.assertEqual(score, 1.0, details)
        self.assertTrue(all(details["checks"].values()))
        self.assertEqual(details["gates_run"],
                         ["early-verdict", "legs", "preflight"])

    def test_missing_answer_scores_zero(self) -> None:
        score, _ = self._score(None)
        self.assertEqual(score, 0.0)

    def test_wrong_binding_leg_scores_zero(self) -> None:
        bad = copy.deepcopy(self.valid_answer)
        bad["binding_leg"] = "prefill"
        score, details = self._score(bad)
        self.assertEqual(score, 0.0)
        self.assertEqual(details["checks"]["binding_leg_is_decode"], 0)

    def test_accepting_off_leg_prefill_arm_scores_zero(self) -> None:
        bad = copy.deepcopy(self.valid_answer)
        bad["arm_decisions"]["arm_prefill_drafter"] = {
            "decision": "launch",
            "reason": "Faster drafter TP",
        }
        score, details = self._score(bad)
        self.assertEqual(score, 0.0)
        self.assertEqual(details["checks"]["rejects_off_leg_prefill_arm"], 0)

    def test_accepting_symmetric_dcp8_scores_zero(self) -> None:
        bad = copy.deepcopy(self.valid_answer)
        bad["arm_decisions"]["arm_sym_dcp8"] = {
            "decision": "launch",
            "reason": "Lower DCP speeds up decode",
        }
        bad["selected_arm"] = "arm_sym_dcp8"
        score, details = self._score(bad)
        self.assertEqual(score, 0.0)
        self.assertEqual(details["checks"]["rejects_symmetric_dcp8"], 0)

    def test_missing_80_row_bucket_cap_scores_zero(self) -> None:
        bad = copy.deepcopy(self.valid_answer)
        bad["arm_decisions"]["arm_cap20_mtp3"] = {
            "decision": "launch",
            "reason": "Higher decode batch cap increases throughput",
        }
        score, details = self._score(bad)
        self.assertEqual(score, 0.0)
        self.assertEqual(details["checks"]["flags_80_row_bucket_cap"], 0)

    def test_failing_to_stop_losing_inflight_arm_scores_zero(self) -> None:
        bad = copy.deepcopy(self.valid_answer)
        bad["inflight_action"] = "watch"
        score, details = self._score(bad)
        self.assertEqual(score, 0.0)
        self.assertEqual(details["checks"]["stops_losing_inflight_arm"], 0)

    def test_right_decisions_with_empty_reasons_score_zero(self) -> None:
        bad = copy.deepcopy(self.valid_answer)
        for arm in ARMS:
            bad["arm_decisions"][arm]["reason"] = ""
        bad["inflight_reason"] = ""
        score, details = self._score(bad)
        self.assertEqual(score, 0.0)
        for check in ("rejects_off_leg_prefill_arm", "rejects_symmetric_dcp8",
                      "flags_80_row_bucket_cap", "stops_losing_inflight_arm"):
            self.assertEqual(details["checks"][check], 0, check)
        self.assertEqual(details["checks"]["selects_asymmetric_dcp_arm"], 1)

    def test_right_decisions_with_wrong_reasons_score_zero(self) -> None:
        bad = copy.deepcopy(self.valid_answer)
        bad["arm_decisions"]["arm_prefill_drafter"]["reason"] = (
            "The prefill drafter change has not been validated on prefill.")
        bad["arm_decisions"]["arm_sym_dcp8"]["reason"] = (
            "Halving DCP width changes the decode collective.")
        bad["arm_decisions"]["arm_cap20_mtp3"]["reason"] = (
            "A larger cap needs a kernel change first.")
        bad["inflight_reason"] = "Throughput and ITL are worse than baseline."
        score, details = self._score(bad)
        self.assertEqual(score, 0.0)
        for check in ("rejects_off_leg_prefill_arm", "rejects_symmetric_dcp8",
                      "flags_80_row_bucket_cap", "stops_losing_inflight_arm"):
            self.assertEqual(details["checks"][check], 0, check)

    def test_handwritten_mechanism_reasons_pass(self) -> None:
        good = copy.deepcopy(self.valid_answer)
        good["arm_decisions"]["arm_prefill_drafter"]["reason"] = (
            "Decode leg is saturated (16/16 running, 4 waiting) while prefill "
            "never queues, so a prefill-only change cannot help.")
        good["arm_decisions"]["arm_sym_dcp8"]["reason"] = (
            "Symmetric DCP=8 halves the prefill KV cache at c=48.")
        good["arm_decisions"]["arm_cap20_mtp3"]["reason"] = (
            "20 x 4 = 80 tokens per step exceeds the largest ROW_BUCKETS "
            "entry (64).")
        good["inflight_reason"] = "-24.8% equal-time tok/s/chip at 10.5 min."
        score, details = self._score(good)
        self.assertEqual(score, 1.0, details)

    def test_right_answer_without_hillclimb_cli_scores_zero(self) -> None:
        score, details = self._score(self.valid_answer, trajectory={})
        self.assertEqual(score, 0.0)
        self.assertEqual(details["checks"]["ran_preflight_or_legs_gate"], 0)
        self.assertEqual(details["checks"]["ran_early_verdict_gate"], 0)
        self.assertFalse(details["trajectory_present"])

    def test_echoed_cli_commands_without_cli_output_score_zero(self) -> None:
        command = "echo hillclimb_cli.py preflight legs early-verdict"
        score, details = self._score(
            self.valid_answer,
            trajectory=_trajectory([(command, command[5:] + "\n")]),
        )
        self.assertEqual(score, 0.0)
        self.assertEqual(details["gates_run"], [])

    def test_early_verdict_gate_is_required(self) -> None:
        calls = [c for c in self.cli_calls if " early-verdict " not in c[0]]
        score, details = self._score(self.valid_answer,
                                     trajectory=_trajectory(calls))
        self.assertEqual(score, 0.0)
        self.assertEqual(details["checks"]["ran_preflight_or_legs_gate"], 1)
        self.assertEqual(details["checks"]["ran_early_verdict_gate"], 0)

    def test_submitting_a_compute_job_scores_zero(self) -> None:
        calls = [
            *self.cli_calls,
            ("tpu-agent-suite compute submit job.yaml", "submitted\n")
        ]
        score, details = self._score(self.valid_answer,
                                     trajectory=_trajectory(calls))
        self.assertEqual(score, 0.0)
        self.assertEqual(details["checks"]["no_compute_job_submitted"], 0)

    def test_compute_run_and_flagged_submit_score_zero(self) -> None:
        for command in (
                "python3 runtime.py compute run --recipe probe.sh",
            ("python3 runtime.py compute --package "
             "tpu-agent-suite-experimental submit --recipe probe.sh"),
        ):
            with self.subTest(command=command):
                calls = [*self.cli_calls, (command, "job-1\n")]
                score, details = self._score(self.valid_answer,
                                             trajectory=_trajectory(calls))
                self.assertEqual(score, 0.0)
                self.assertEqual(details["checks"]["no_compute_job_submitted"],
                                 0)

    def test_forged_gate_markers_score_zero(self) -> None:
        findings = '{"has_errors": false, "findings": []}'
        forged = [
            (f"echo '{findings}'  # preflight", findings + "\n"),
            ("printf 'early_verdict: STOP\\n'  # early-verdict",
             "early_verdict: STOP\n"),
            # Names the CLI and prints its generic markers, but no value that
            # only this task's fixtures produce.
            (f"echo hillclimb_cli.py preflight; echo '{findings}'",
             f"hillclimb_cli.py preflight\n{findings}\n"),
            ("echo hillclimb_cli.py early-verdict; echo 'early_verdict: STOP'",
             "hillclimb_cli.py early-verdict\nearly_verdict: STOP\n"),
        ]
        score, details = self._score(self.valid_answer,
                                     trajectory=_trajectory(forged))
        self.assertEqual(score, 0.0)
        self.assertEqual(details["gates_run"], [])

    def test_gate_output_from_another_script_scores_zero(self) -> None:
        renamed = [(command.replace("hillclimb_cli.py", "/tmp/gates.py"), out)
                   for command, out in self.cli_calls]
        score, details = self._score(self.valid_answer,
                                     trajectory=_trajectory(renamed))
        self.assertEqual(score, 0.0)
        self.assertEqual(details["gates_run"], [])

    def test_malformed_trajectory_scores_zero(self) -> None:
        unhashable_ids = json.dumps({
            "steps": [{
                "tool_calls": [{
                    "function_name": "Bash",
                    "tool_call_id": ["unhashable"],
                    "arguments": {
                        "command": "echo hi"
                    },
                }],
                "observation": {
                    "results": [{
                        "source_call_id": ["unhashable"],
                        "content": "hi",
                    }]
                },
            }]
        })
        for raw in ("{not json", "[]", '{"steps": "none"}',
                    '{"steps": [1, "x", {"tool_calls": "bad"}]}',
                    unhashable_ids):
            with self.subTest(raw=raw):
                score, details = self._score(self.valid_answer,
                                             raw_trajectory=raw)
                self.assertEqual(score, 0.0)
                self.assertTrue(details["trajectory_present"])
                self.assertEqual(details["gates_run"], [])

    def test_many_flag_non_submit_compute_command_stays_fast_and_passes(
            self) -> None:
        many_flags = " ".join(f"--flag-{i} val{i}" for i in range(40))
        calls = [
            *self.cli_calls,
            (f"python3 runtime.py compute {many_flags} job status", "ok\n"),
        ]
        score, details = self._score(self.valid_answer,
                                     trajectory=_trajectory(calls))
        self.assertEqual(score, 1.0, details)

    def test_text_output_of_the_gates_counts(self) -> None:
        calls: list = []
        metrics = str(WORK_DIR / "metrics" / "server_metrics_export.csv")
        _run_cli_text(calls, "legs", "--server-metrics", metrics)
        _run_cli_text(calls, "preflight", "--knob", "DCP_SIZE=8")
        _run_cli_text(
            calls,
            "early-verdict",
            "--baseline-log",
            str(WORK_DIR / "telemetry" / "baseline_aiperf.txt"),
            "--arm-log",
            str(WORK_DIR / "telemetry" / "inflight_arm_aiperf.txt"),
        )
        score, details = self._score(self.valid_answer,
                                     trajectory=_trajectory(calls))
        self.assertEqual(score, 1.0, details)
        self.assertEqual(details["gates_run"],
                         ["early-verdict", "legs", "preflight"])


if __name__ == "__main__":
    unittest.main()
