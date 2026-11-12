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
"""ici-mesh-collective-portfolio's verifier, run for real on synthetic trials.

The answer can be worked out by hand from the five ops in op_profile.json, so
the cases that matter are a right answer with no run of the skill behind it,
an echoed command that only looks like one, and a right answer that never
reached the file the instruction names. All of them scored a full reward
before these cases existed.

The trajectories are hand-written ATIF steps with the fields the verifier
reads: Claude Code's Bash and Antigravity's run_command calls, the failure
flags each harness sets on a result, and the result's content. Where a call
ran the skill's CLI, its content is the CLI's real output on the task's
fixture.
"""
from __future__ import annotations

import functools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TASK = (REPO / "eval" / "datasets" / "ici-mesh-sharding" /
        "ici-mesh-collective-portfolio")
VERIFIER = TASK / "tests" / "test.sh"
SKILL_CLI = REPO / "package" / "scripts" / "ici-mesh-sharding" / "cli.py"

SKILL_DIR = ("~/.tpu-agent-suite/packages/tpu-agent-suite-experimental/"
             "scripts/ici-mesh-sharding")
CLI = f"python3 {SKILL_DIR}/cli.py"
MESH_JSON = '{"data":1,"fsdp":4,"tensor":2,"sequence":1,"expert":8}'
MESH = f"'{MESH_JSON}'"
BREAKDOWN_ARGS = ("--ops-json /workspace/work/op_profile.json"
                  f" --mesh {MESH} --output-format json")
RANK_ARGS = ("--ops-json /workspace/work/op_profile.json"
             f" --mesh {MESH} --topology 4x4x4 --batch-size 128"
             " --sequence-length 8192 --num-heads 32 --num-kv-heads 8")
BREAKDOWN = f"{CLI} breakdown {BREAKDOWN_ARGS}"
RANK = f"{CLI} rank-portfolio {RANK_ARGS} --output-format json"


@functools.cache
def _cli_output(*args: str) -> str:
    """What the skill's CLI prints for `args` on the task's fixture."""
    work = TASK / "environment" / "work"
    argv = [
        part.replace("/workspace/work", str(work)).replace(MESH, MESH_JSON)
        for part in args
    ]
    return subprocess.run(
        [sys.executable, str(SKILL_CLI), *argv],
        check=True,
        capture_output=True,
        text=True).stdout


def _rank_output(fmt: str = "json") -> str:
    return _cli_output("rank-portfolio", *RANK_ARGS.split(), "--output-format",
                       fmt)


def _breakdown_output() -> str:
    return _cli_output("breakdown", *BREAKDOWN_ARGS.split())


def _proposal(data: int, fsdp: int, tensor: int, sequence: int,
              expert: int) -> dict:
    return {
        "mesh":
        dict(data=data,
             fsdp=fsdp,
             tensor=tensor,
             sequence=sequence,
             expert=expert)
    }


# What `cli.py breakdown`, `check-topology` and `rank-portfolio` report for
# the task's fixture; the meshes are three of rank-portfolio's proposals.
ANSWER = dict(
    primary_bottleneck_axis="expert",
    primary_bottleneck_role="MoE routing",
    collective_pct=57.0,
    baseline_requires_split_physical_axes=True,
    proposals=[
        _proposal(1, 4, 4, 1, 4),
        _proposal(1, 4, 1, 1, 16),
        _proposal(1, 8, 2, 1, 4),
    ],
)


def _bash(command: str,
          *,
          output: str = "",
          is_error: bool | None = None) -> dict:
    """A Claude Code Bash call and its result."""
    extra = None if is_error is None else {"tool_result_is_error": is_error}
    return {
        "tool_calls": [{
            "tool_call_id": "toolu_1",
            "function_name": "Bash",
            "arguments": {
                "command": command
            },
        }],
        "observation": {
            "results": [{
                "source_call_id": "toolu_1",
                "content": output,
                "extra": extra
            }]
        },
    }


def _run_command(command: str,
                 *,
                 output: str = "",
                 status: str = "DONE") -> dict:
    """An Antigravity run_command call and its result."""
    return {
        "tool_calls": [{
            "tool_call_id": "antigravity-7-0",
            "function_name": "run_command",
            "arguments": {
                "CommandLine": command,
                "Cwd": "/workspace"
            },
        }],
        "observation": {
            "results": [{
                "source_call_id": "antigravity-7-0",
                "content": output,
                "extra": {
                    "status": status
                },
            }]
        },
    }


def _call(name: str, args: dict) -> dict:
    """Any other tool call, with a result."""
    return {
        "tool_calls": [{
            "tool_call_id": "toolu_2",
            "function_name": name,
            "arguments": args
        }],
        "observation": {
            "results": [{
                "source_call_id": "toolu_2",
                "content": "done"
            }]
        },
    }


def _message(text: str) -> dict:
    return {"source": "agent", "message": text}


def _used_the_skill() -> tuple[dict, ...]:
    return (_bash(BREAKDOWN, output=_breakdown_output()),
            _bash(RANK, output=_rank_output()))


_USED_THE_SKILL = object()


class IciMeshPortfolioVerifierTest(unittest.TestCase):
    """What the verifier scores, driven through the real script."""

    def score(self,
              *,
              answer: object = ANSWER,
              steps: object = _USED_THE_SKILL,
              trajectory_dir: str = "agent") -> dict:
        """Runs the verifier over one synthetic trial; returns its details.

        `steps` None writes no trajectory at all.
        """
        if steps is _USED_THE_SKILL:
            steps = _used_the_skill()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent, verifier, work = (root / "agent", root / "verifier",
                                     root / "work")
            for d in (agent, verifier, work):
                d.mkdir()
            if answer is not None:
                (work / "portfolio_recommendation.json").write_text(
                    answer if isinstance(answer, str) else json.dumps(answer),
                    encoding="utf-8")
            if steps is not None:
                where = agent if trajectory_dir == "agent" else root
                (where / "trajectory.json").write_text(json.dumps({
                    "schema_version":
                    "ATIF-v1.7",
                    "steps":
                    list(steps)
                }),
                                                       encoding="utf-8")
            done = subprocess.run(
                ["bash", str(VERIFIER)],
                env={
                    "PATH": "/usr/bin:/bin:/usr/local/bin",
                    "LOGS_DIR": str(verifier),
                    "WORK_DIR": str(work),
                    "AGENT_LOGS_DIR": str(agent),
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

    def test_a_run_that_ranked_with_the_skill_scores(self) -> None:
        got = self.score()
        self.assertEqual(got["reward"], 1.0, got)
        self.assertEqual(got["cli_subcommands"],
                         ["breakdown", "rank-portfolio"])

    def test_antigravity_run_command_counts(self) -> None:
        got = self.score(steps=(_run_command(RANK, output=_rank_output()), ))
        self.assertEqual(got["reward"], 1.0, got)

    def test_a_cwd_relative_run_with_its_output_counts(self) -> None:
        """`cd` into the scripts in one call, then run `cli.py` bare."""
        got = self.score(steps=(
            _bash(f"cd {SKILL_DIR}"),
            _bash(f"python3 cli.py rank-portfolio {RANK_ARGS}",
                  output=_rank_output("table")),
        ))
        self.assertEqual(got["checks"]["ran_rank_portfolio"], 1)
        self.assertEqual(got["reward"], 1.0, got)

    def test_an_echoed_command_without_cli_output_fails(self) -> None:
        got = self.score(steps=(_bash(f"echo {RANK}", output=RANK[8:]), ))
        self.assertEqual(got["cli_subcommands"], ["rank-portfolio"])
        self.assertEqual(got["checks"]["ran_rank_portfolio"], 0)
        self.assertEqual(got["reward"], 0.0)

    def test_output_keys_typed_into_the_command_do_not_count(self) -> None:
        echoed = 'cli.py rank-portfolio "hypotheses" "estimated_gain_pct"'
        got = self.score(steps=(_bash(f"echo '{echoed}'", output=echoed), ))
        self.assertEqual(got["checks"]["ran_rank_portfolio"], 0)
        self.assertEqual(got["reward"], 0.0)

    def test_right_json_without_the_skill_fails(self) -> None:
        """The answer worked out by hand: read the profile, wrote the file."""
        got = self.score(steps=(
            _call("Read", {"file_path": "/workspace/work/op_profile.json"}),
            _call(
                "Write", {
                    "file_path":
                    "/workspace/work/portfolio_recommendation.json",
                    "content": json.dumps(ANSWER)
                }),
        ))
        self.assertEqual(got["checks"]["ran_rank_portfolio"], 0)
        self.assertEqual(got["reward"], 0.0)

    def test_loading_the_skill_without_running_it_fails(self) -> None:
        """Reading SKILL.md, which quotes the command, runs nothing."""
        got = self.score(steps=(
            _call("Skill", {"skill": "ici-mesh-sharding"}),
            _call("Read", {"file_path": f"{SKILL_DIR}/cli.py"}),
        ))
        self.assertEqual(got["checks"]["ran_rank_portfolio"], 0)
        self.assertEqual(got["reward"], 0.0)

    def test_breakdown_alone_is_not_a_ranked_portfolio(self) -> None:
        got = self.score(
            steps=(_bash(BREAKDOWN, output=_breakdown_output()), ))
        self.assertEqual(got["checks"]["ran_rank_portfolio"], 0)
        self.assertEqual(got["reward"], 0.0)

    def test_a_failed_or_help_only_run_does_not_count(self) -> None:
        for steps in (
            (_bash(RANK, output=_rank_output(), is_error=True), ),
            (_run_command(RANK, output=_rank_output(), status="ERROR"), ),
            (_bash(f"{CLI} rank-portfolio --help",
                   output=_cli_output("rank-portfolio", "--help")), ),
        ):
            with self.subTest(steps=steps):
                got = self.score(steps=steps)
                self.assertEqual(got["checks"]["ran_rank_portfolio"], 0)
                self.assertEqual(got["reward"], 0.0)

    def test_an_answer_only_in_the_final_message_fails(self) -> None:
        """The JSON in chat is reported, not scored: the file is the answer."""
        got = self.score(answer=None,
                         steps=_used_the_skill() +
                         (_message(f"```json\n{json.dumps(ANSWER)}\n```"), ))
        self.assertEqual(got["checks"]["wrote_recommendation_json"], 0)
        self.assertEqual(got["checks"]["accurate_collective_pct"], 0)
        self.assertTrue(got["message_json_found"])
        self.assertEqual(got["reward"], 0.0)

    def test_an_unreadable_answer_file_fails(self) -> None:
        got = self.score(answer="{not json")
        self.assertEqual(got["checks"]["wrote_recommendation_json"], 0)
        self.assertIsNotNone(got["error"])
        self.assertEqual(got["reward"], 0.0)

    def test_without_a_trajectory_the_trajectory_checks_fail_closed(
            self) -> None:
        got = self.score(steps=None)
        self.assertFalse(got["trajectory_present"])
        self.assertEqual(got["checks"]["ran_rank_portfolio"], 0)
        self.assertEqual(got["checks"]["no_compute_job_submitted"], 0)
        self.assertEqual(got["checks"]["wrote_recommendation_json"], 1)
        self.assertEqual(got["reward"], 0.0)

    def test_the_trajectory_beside_the_agent_logs_is_read(self) -> None:
        got = self.score(trajectory_dir="logs")
        self.assertTrue(got["trajectory_present"])
        self.assertEqual(got["reward"], 1.0, got)

    def test_submitting_a_compute_job_fails(self) -> None:
        got = self.score(steps=_used_the_skill() +
                         (_bash("compute run --code probe.py --timeout 60"), ))
        self.assertEqual(got["checks"]["no_compute_job_submitted"], 0)
        self.assertEqual(got["reward"], 0.0)

    def test_a_wrong_answer_fails_even_with_the_skill(self) -> None:
        for key, value in (
            ("collective_pct", 38.0),
            ("baseline_requires_split_physical_axes", False),
            ("primary_bottleneck_axis", "fsdp"),
        ):
            with self.subTest(key=key):
                got = self.score(answer=dict(ANSWER, **{key: value}))
                self.assertEqual(got["reward"], 0.0)


if __name__ == "__main__":
    unittest.main()
