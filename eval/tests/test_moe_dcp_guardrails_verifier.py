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
"""The moe-dcp-guardrails verifier, run for real against synthetic trials.

The task asks for the serving optimization heuristics' answer, and the
knowledge base states the same facts, so a run can write the right
answer.json without running the analyzer. These cases pin that such a run
scores 0: detector IDs count only from one plain run of the installed suite's
analyze.py whose output carries the analyzer's own lines, or from a
findings.json that a tpu-diagnostics delegation's result names. Echoed IDs,
look-alike scripts, chained commands and files the agent wrote itself do not
count.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

VERIFIER = (Path(__file__).resolve().parents[1] / "datasets" /
            "onboarding-routing" /
            "optimization-heuristics-moe-dcp-guardrails" / "tests" / "test.sh")

D52 = "D52_moe_tp_decode_row_limit"
D18 = "D18_attn_indexer"
IDS = f"{D52} {D18}"
ANSWER = {
    "detector_ids": [D18, D52],
    "decode_enable_expert_parallel":
    True,
    "decode_dcp_size":
    8,
    "prefill_dcp_size":
    16,
    "allow_tp1_mesh":
    False,
    "answer": ("Add --enable-expert-parallel on the decode engine at c=20;"
               " set DECODE_DCP_SIZE=8 and keep PREFILL_DCP_SIZE=16."),
    "reference":
    "findings.json",
}
SUITE = "/root/.tpu-agent-suite/packages/suite"
SCRIPT = f"{SUITE}/scripts/optimization-heuristics/analyze.py"
# The analyzer as the compiled tpu-diagnostics agent spells it.
COMPILED_SCRIPT = ("~/.tpu-agent-suite/packages/tpu-agent-suite@0.7.0/scripts/"
                   "optimization-heuristics/analyze.py")
EVENTS = ("--events /workspace/profile/events/decode-c12-48tok-dcp16.parquet,"
          "role=decode,point=8192:8192:12 --events"
          " /workspace/profile/events/decode-c20-80tok-dcp16.parquet,"
          "role=decode,point=8192:8192:20")
ANALYZE = (
    f"python3 {SCRIPT} --deployments"
    f" /workspace/inference/deployments.json {EVENTS} --out-dir {{out}}")
# What analyze.py prints: one line per profile, then where it wrote.
PRINTED = (f"decode_c12_48tok: D2_padding_waste, {D18}\n"
           f"decode_c20_80tok: D2_padding_waste, {D52}\n"
           "Wrote /tmp/oh/findings.json and report.md\n")
RUN_FINDINGS = ("run-1/artifacts/diagnostics/optimization-heuristics/"
                "findings.json")
FINDINGS = {
    "serve_command":
    "vllm serve zai-org/GLM-5-FP8",
    "profiles": [
        {
            "label": "decode_c12_48tok",
            "result": {
                "findings": [{
                    "detector_id": D18
                }]
            }
        },
        {
            "label": "decode_c20_80tok",
            "result": {
                "findings": [{
                    "detector_id": D52
                }]
            }
        },
    ],
}


def _call(name: str,
          args: dict,
          content: str = "",
          error: bool = False) -> dict:
    """One ATIF step: a tool call and its result."""
    result = {"source_call_id": "call-1", "content": content}
    if error:
        result["extra"] = {"tool_result_is_error": True}
    return {
        "tool_calls": [{
            "tool_call_id": "call-1",
            "function_name": name,
            "arguments": args
        }],
        "observation": {
            "results": [result]
        },
    }


def _bash(command: str, content: str = PRINTED) -> dict:
    """A Claude Code Bash call and what it printed."""
    return _call("Bash", {"command": command}, content)


class MoeDcpGuardrailsVerifierTest(unittest.TestCase):
    """What the verifier scores, driven through the real script."""

    def score(self,
              steps: list | None,
              answer: dict | None = None,
              suite_findings: bool = False,
              workspace_findings: bool = False,
              out_dir_findings: str = "",
              write_answer: bool = True) -> dict:
        """Runs the verifier over one synthetic trial; returns its details."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent, verifier, runs, workspace = (root / "agent",
                                                root / "verifier",
                                                root / "runs",
                                                root / "workspace")
            for d in (agent, verifier, runs, workspace):
                d.mkdir()
            if steps is not None:
                (agent / "trajectory.json").write_text(json.dumps(
                    {"steps": steps}),
                                                       encoding="utf-8")
            if write_answer:
                (root / "answer.json").write_text(json.dumps(answer or ANSWER),
                                                  encoding="utf-8")
            payload = json.dumps(FINDINGS)
            if suite_findings:
                out = runs / RUN_FINDINGS
                out.parent.mkdir(parents=True)
                out.write_text(payload, encoding="utf-8")
            if workspace_findings:
                (workspace / "findings.json").write_text(payload,
                                                         encoding="utf-8")
            if out_dir_findings:
                out = workspace / out_dir_findings
                out.mkdir(parents=True)
                (out / "findings.json").write_text(payload, encoding="utf-8")
            done = subprocess.run(
                ["bash", str(VERIFIER)],
                env={
                    "PATH": "/usr/bin:/bin:/usr/local/bin",
                    "LOGS_DIR": str(verifier),
                    "ANSWER_JSON_PATH": str(root / "answer.json"),
                    "AGENT_LOGS_DIR": str(agent),
                    "TPU_AGENT_SUITE_RUNS": str(runs),
                    "WORKSPACE_DIR": str(workspace),
                },
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            details = json.loads(
                (verifier / "reward-details.json").read_text())
            details["reward"] = json.loads(
                (verifier / "reward.json").read_text())["reward"]
            return details

    def test_analyzer_run_in_the_trajectory_scores(self) -> None:
        got = self.score([_bash(ANALYZE.format(out="/tmp/oh"))])
        self.assertEqual(got["reward"], 1.0, got)
        self.assertEqual(got["analyzer_detector_ids"],
                         sorted([D18, D52, "D2_padding_waste"]))

    def test_plain_analyzer_forms_score(self) -> None:
        """Prefixes, a leading cd, continuations and redirections are fine."""
        forms = {
            "antigravity run_command":
            _call("run_command", {"CommandLine": ANALYZE.format(out="oh")},
                  PRINTED),
            "variable prefix":
            _bash(
                f"PYTHONPATH=/x python3 {SCRIPT} {EVENTS} --out-dir /tmp/oh"),
            "env prefix":
            _bash(f"env PYTHONPATH=/x python3 {SCRIPT} {EVENTS}"
                  " --out-dir /tmp/oh"),
            "absolute python":
            _bash(f"/usr/bin/python3 {SCRIPT} {EVENTS} --out-dir /tmp/oh"),
            "timeout prefix":
            _bash(f"timeout 600 python3 {SCRIPT} {EVENTS} --out-dir /tmp/oh"),
            "cd on its own line":
            _bash(f"cd /workspace\npython3 {SCRIPT} {EVENTS} --out-dir oh"),
            "cd and a redirection":
            _bash(f"cd /workspace && python3 {SCRIPT} {EVENTS} --out-dir oh"
                  " 2>&1"),
            "compiled path with line continuations":
            _bash(
                f"python3 {COMPILED_SCRIPT} \\\n"
                "    --deployments /workspace/inference/deployments.json \\\n"
                f"    {EVENTS} \\\n"
                "    --out-dir /tmp/oh"),
            "relative script below a cd into the package":
            _bash(
                "cd ~/.tpu-agent-suite/packages/tpu-agent-suite@0.7.0/scripts"
                f" && python3 optimization-heuristics/analyze.py {EVENTS}"
                " --out-dir /tmp/oh"),
        }
        for name, step in forms.items():
            with self.subTest(form=name):
                got = self.score([step])
                self.assertEqual(got["checks"]["analyzer_ran"], 1, got)
                self.assertEqual(got["reward"], 1.0, got)

    def test_spoofed_analyzer_evidence_scores_zero(self) -> None:
        """Right answer.json, but nothing shows the analyzer printed it."""
        usage = ("usage: analyze.py [-h] --events EVENTS (--deployments"
                 " DEPLOYMENTS | --serve-command SERVE_COMMAND)\n")
        spoofs = {
            "echo with a separator inside quotes":
            ([_bash(f"echo '; python3 {SCRIPT} {EVENTS} {IDS}'", IDS)], {}),
            "short-circuited analyzer, then echo":
            ([_bash(f"true || python3 {SCRIPT} {EVENTS}; echo {IDS}")], {}),
            "failing analyzer, then echo": ([
                _bash(
                    f"python3 {SCRIPT} --events /nonexistent.parquet;"
                    f" echo {IDS}",
                    "Traceback (most recent call last):\n" + PRINTED)
            ], {}),
            "analyzer help, then a knowledge-base page": ([
                _bash(
                    f"python3 {SCRIPT} --help --events;"
                    " cat /root/kb/vllm-serving-heuristics.md",
                    usage + PRINTED)
            ], {}),
            "look-alike script outside the suite": ([
                _bash("python3 /tmp/x/optimization-heuristics/analyze.py"
                      f" {EVENTS} --out-dir /tmp/oh")
            ], {}),
            "path that climbs out of the suite": ([
                _bash(f"python3 {SUITE}/../../../../tmp/x/scripts/"
                      f"optimization-heuristics/analyze.py {EVENTS}")
            ], {}),
            "plain analyzer call without its Wrote line":
            ([_bash(f"python3 {SCRIPT} --help --events", usage + IDS)], {}),
            "agent-written --out-dir behind a failing call": ([
                _bash(
                    f"python3 {SCRIPT} --events /nonexistent --out-dir fake"
                    " || true", usage + "error: one of the arguments"
                    " --deployments --serve-command is required\n")
            ], {
                "out_dir_findings":
                "fake"
            }),
            "agent-written run findings behind a no-op delegation":
            ([_call("Agent", {"subagent_type": "tpu-diagnostics"}, "hi")], {
                "suite_findings": True
            }),
        }
        for name, (steps, kwargs) in spoofs.items():
            with self.subTest(spoof=name):
                got = self.score(steps, **kwargs)
                self.assertEqual(got["checks"]["analyzer_ran"], 0, got)
                self.assertEqual(got["reward"], 0.0, got)

    def test_out_dir_findings_without_the_analyzer_output_do_not_count(
            self) -> None:
        got = self.score(
            [_call("run_command", {"CommandLine": ANALYZE.format(out="oh")})],
            out_dir_findings="oh")
        self.assertEqual(got["checks"]["analyzer_ran"], 0)
        self.assertEqual(got["reward"], 0.0)

    def test_delegation_naming_its_findings_scores(self) -> None:
        named = f"findings.json: /root/.tpu-agent-suite/runs/{RUN_FINDINGS}"
        delegations = {
            "claude code":
            _call("Agent", {"subagent_type": "tpu-diagnostics"}, named),
            "antigravity":
            _call("invoke_subagent",
                  {"Subagents": [{
                      "TypeName": "tpu-diagnostics"
                  }]}, named),
        }
        for name, step in delegations.items():
            with self.subTest(harness=name):
                got = self.score([step], suite_findings=True)
                self.assertEqual(got["reward"], 1.0, got)

    def test_right_answer_without_the_analyzer_scores_zero(self) -> None:
        """The self-reported case: a knowledge-base lookup, then answer.json."""
        lookup = _bash(
            "cat /root/.tpu-agent-suite/kb/vllm-serving-heuristics.md",
            f"{D52} ... {D18} ... DECODE_DCP_SIZE=8")
        got = self.score([lookup])
        self.assertEqual(got["checks"]["analyzer_ran"], 0)
        self.assertEqual(got["checks"]["has_d52"], 0)
        self.assertEqual(got["reward"], 0.0)
        echo_spoof = _bash(
            f"echo 'optimization-heuristics/analyze.py --events {IDS}'", IDS)
        got_spoof = self.score([echo_spoof])
        self.assertEqual(got_spoof["checks"]["analyzer_ran"], 0)
        self.assertEqual(got_spoof["reward"], 0.0)

    def test_agent_written_findings_do_not_count(self) -> None:
        got = self.score([], workspace_findings=True)
        self.assertEqual(got["reward"], 0.0)
        got = self.score([], suite_findings=True)
        self.assertEqual(got["checks"]["analyzer_ran"], 0)
        self.assertEqual(got["reward"], 0.0)

    def test_missing_trajectory_fails_closed(self) -> None:
        got = self.score(None, suite_findings=True, workspace_findings=True)
        self.assertEqual(got["trajectory_present"], False)
        self.assertEqual(got["checks"]["analyzer_ran"], 0)
        self.assertEqual(got["reward"], 0.0)

    def test_malformed_trajectory_scores_zero(self) -> None:
        """Entries of the wrong JSON type are skipped instead of crashing."""
        not_steps = ["not a step", ["not", "a", "step"]]
        bad_call = {
            "tool_call_id": ["call-1"],
            "function_name": "Bash",
            "arguments": ANALYZE.format(out="/tmp/oh")
        }
        bad_fields = [
            {
                "tool_calls": 7,
                "observation": "done"
            },
            {
                "tool_calls": "Bash",
                "observation": {
                    "results": 5
                }
            },
            {
                "tool_calls": ["not a call", bad_call],
                "observation": {
                    "results": [
                        "not a result", {
                            "source_call_id": ["call-1"],
                            "content": PRINTED
                        }
                    ]
                },
            },
            {
                "tool_calls": [{
                    "tool_call_id": "call-1",
                    "function_name": "invoke_subagent",
                    "arguments": {
                        "Subagents": 3
                    }
                }],
                "observation": {
                    "results": [{
                        "source_call_id": "call-1",
                        "extra": ["ERROR"],
                        "content": 5
                    }]
                },
            },
        ]
        for name, steps in (("a string and a list as steps", not_steps),
                            ("fields of the wrong type", bad_fields)):
            # Without answer.json the verifier also scans the steps for one.
            for write_answer in (True, False):
                with self.subTest(trajectory=name, answer_json=write_answer):
                    got = self.score(steps, write_answer=write_answer)
                    self.assertEqual(got["checks"]["analyzer_ran"], 0, got)
                    self.assertEqual(got["reward"], 0.0, got)
        # Next to a real analyzer run, the same entries do not hide it.
        got = self.score(
            [*not_steps, *bad_fields,
             _bash(ANALYZE.format(out="/tmp/oh"))])
        self.assertEqual(got["reward"], 1.0, got)

    def test_failed_analyzer_call_does_not_count(self) -> None:
        got = self.score([
            _call("Bash", {"command": ANALYZE.format(out="/tmp/oh")},
                  PRINTED,
                  error=True)
        ])
        self.assertEqual(got["reward"], 0.0)

    def test_answer_must_state_decode_8_and_prefill_16(self) -> None:
        steps = [_bash(ANALYZE.format(out="/tmp/oh"))]
        vague = dict(ANSWER,
                     answer=("Enable expert parallel on decode; leave the"
                             " prefill engine as it is."))
        got = self.score(steps, answer=vague)
        self.assertEqual(got["checks"]["has_answer_and_reference"], 0)
        self.assertEqual(got["reward"], 0.0)
        reversed_sizes = dict(ANSWER,
                              answer=("Enable expert parallel; DCP 8 on the"
                                      " decode engine, DCP 16 for prefill."))
        self.assertEqual(
            self.score(steps, answer=reversed_sizes)["reward"], 1.0)

    def test_compute_run_and_flagged_submit_score_zero(self) -> None:
        analyzer = _bash(ANALYZE.format(out="/tmp/oh"))
        for command in (
                "compute run --code /workspace/probe.py --timeout 300",
                "compute --package x submit --recipe probe.py",
                # How compiled agents spell `compute run`.
                "compute --package tpu-agent-suite@0.7.0 run --code probe.py",
                "kubectl apply -f job.yaml",
        ):
            with self.subTest(command=command):
                got = self.score([analyzer, _bash(command, "job-1\n")])
                self.assertEqual(got["checks"]["no_compute_job_submitted"], 0,
                                 got)
                self.assertEqual(got["reward"], 0.0, got)
        # Reading a job's status or the compute config submits nothing.
        for command in (
                "compute status --job-id 12",
                "compute --package tpu-agent-suite@0.7.0 inspect --json",
        ):
            with self.subTest(command=command):
                got = self.score([analyzer, _bash(command, "{}\n")])
                self.assertEqual(got["reward"], 1.0, got)


if __name__ == "__main__":
    unittest.main()
