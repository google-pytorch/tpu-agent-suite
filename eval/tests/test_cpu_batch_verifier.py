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
"""cpu-batch's verifier, run for real against synthetic trial logs.

The task exists to catch an agent that answers the batch-size question itself,
so the cases that matter are the ones where a run looks right and is not: the
sizes the user never chose, a placeholder where a measurement should be, and
a conversation with no question in it. Each of those scored a full reward
before these fixtures existed.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path

VERIFIER = (Path(__file__).resolve().parents[1] / "datasets" /
            "tpu-hello-world" / "cpu-batch" / "tests" / "test.sh")

TABLE = ("| device_type | cpu-simulated |\n"
         "| batch_mode | range |\n"
         "| batch_sizes | 1,2,3,4 |\n"
         "| max_abs_diff | 2.43e-05 |\n"
         "| tolerance | 1e-3 |\n")


def _turn(role: str, text: str) -> dict:
    return {role: {"content": [{"Text": text}]}}


def _bridge(*turns: dict) -> dict:
    """An acpx export: the conversation, then the raw JSON-RPC log.

    Both halves carry the same prompts. Counting them together is what let one
    user turn read as two.
    """
    prompts = [{"method": "session/prompt"} for t in turns if "User" in t]
    return {"session": {"messages": list(turns)}, "jsonrpc_log": prompts}


ASKED = _bridge(
    _turn("User", "Run the task. | device_type | |"),
    _turn("Agent", "Constant or a range, and what value?"),
    _turn("User", "A range: 1..4."),
    _turn("Agent", TABLE),
)
NEVER_ASKED = _bridge(
    _turn("User", "Run the task. | device_type | |"),
    _turn("Agent", TABLE),
)
ONLY_ACKNOWLEDGED = _bridge(
    _turn("User", "Run the task. | device_type | |"),
    _turn("Agent", "Working on it now. No questions."),
    _turn("User", "ok, thanks"),
    _turn("Agent", TABLE),
)
ASKED_THEN_EXPORTED = _bridge(
    _turn("User", "Run the task. | device_type | |"),
    _turn("Agent", "Constant or a range, and what value?"),
    _turn("User", "A range: 1..4."),
    _turn("Agent", TABLE),
    _turn("User", "Before we finish: export the trajectory."),
    _turn("Agent", "Exported."),
)


class CpuBatchVerifierTest(unittest.TestCase):
    """What the verifier scores, driven through the real script."""

    def score(self,
              table: str = TABLE,
              bridge: dict | None = ASKED,
              agent_log: str = "",
              result_json: object = None) -> dict:
        """Run the verifier over one synthetic trial and read its reward."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent, verifier, user = (root / "agent", root / "verifier",
                                     root / "user-agent")
            for d in (agent, verifier, user):
                d.mkdir()
            (agent / "reply.txt").write_text(table + agent_log,
                                             encoding="utf-8")
            if bridge is not None:
                # The conversation holds the agent's final table, so a case
                # that varies the table has to vary it there too.
                payload = json.dumps(bridge, indent=1)
                if table != TABLE:
                    payload = payload.replace(
                        json.dumps(TABLE)[1:-1],
                        json.dumps(table)[1:-1])
                (agent / "bridge-trajectory.json").write_text(payload,
                                                              encoding="utf-8")
            scored = root / "result.json"
            if result_json is not None:
                scored.write_text(result_json if isinstance(result_json, str)
                                  else json.dumps(result_json),
                                  encoding="utf-8")
            done = subprocess.run(
                ["bash", str(VERIFIER)],
                env={
                    "PATH": "/usr/bin:/bin:/usr/local/bin",
                    "LOGS_DIR": str(verifier),
                    "AGENT_LOGS_DIR": str(agent),
                    "USER_AGENT_LOGS_DIR": str(user),
                    "RESULT_JSON": str(scored),
                },
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            return json.loads((verifier / "reward.json").read_text())

    def test_a_run_that_asked_and_complied_scores(self) -> None:
        self.assertEqual(self.score()["reward"], 1)

    def test_the_closing_request_does_not_make_a_run_an_ask(self) -> None:
        """The simulated user keeps talking after the table; that is not an ask."""
        self.assertEqual(self.score(bridge=ASKED_THEN_EXPORTED)["reward"], 1)

    def test_an_agent_that_never_asked_fails(self) -> None:
        got = self.score(bridge=NEVER_ASKED)
        self.assertEqual(got["asked_the_user"], 0)
        self.assertEqual(got["reward"], 0)

    def test_the_protocol_log_is_not_a_second_turn(self) -> None:
        """One prompt, rendered twice by acpx, is still one turn.

        The table reaches only the JSON-RPC half here, which is what put the
        conversation's single `"User"` entry and its `session/prompt` echo
        both "before the table" for a line-ordered reader.
        """
        bridge = {
            "session": {
                "messages": [_turn("User", "Run the task. | device_type | |")]
            },
            "jsonrpc_log": [{
                "method": "session/prompt"
            }, {
                "note": TABLE
            }],
        }
        got = self.score(bridge=bridge)
        self.assertEqual(got["asked_the_user"], 0)
        self.assertEqual(got["reward"], 0)

    def test_an_acknowledgement_is_not_an_ask(self) -> None:
        got = self.score(bridge=ONLY_ACKNOWLEDGED)
        self.assertEqual(got["asked_the_user"], 0)

    def test_a_tool_the_log_only_names_is_not_an_ask(self) -> None:
        """A tool inventory mentions AskUserQuestion; nothing called it."""
        got = self.score(bridge=NEVER_ASKED,
                         agent_log='\ntools: ["Read","AskUserQuestion"]\n')
        self.assertEqual(got["asked_the_user"], 0)

    def test_needs_input_named_in_prose_is_not_an_ask(self) -> None:
        """Several agent definitions discuss `needs_input`; reading one is
        not asking. A doc that quotes the literal result JSON is matched as
        a result -- shape is all the log preserves."""
        doc = "Return `needs_input` and say what the caller must supply.\n"
        got = self.score(bridge=NEVER_ASKED, agent_log="\n" + doc)
        self.assertEqual(got["asked_the_user"], 0)

    def test_a_tool_schema_naming_the_ask_tool_is_not_an_ask(self) -> None:
        """A logged schema lists the tool; listing is not calling."""
        schema = '{"tools": ["Read", "Bash", "AskUserQuestion"]}'
        got = self.score(bridge=NEVER_ASKED, agent_log="\n" + schema + "\n")
        self.assertEqual(got["asked_the_user"], 0)

    def test_a_tool_schema_definition_is_not_an_ask(self) -> None:
        """A full tool declaration carries name=AskUserQuestion and input_schema."""
        schema = json.dumps({
            "name": "AskUserQuestion",
            "description": "Ask the user a question.",
            "input_schema": {
                "type": "object"
            },
        })
        got = self.score(bridge=NEVER_ASKED, agent_log="\n" + schema + "\n")
        self.assertEqual(got["asked_the_user"], 0)

    def test_a_read_tool_result_quoting_needs_input_json_is_not_an_ask(
            self) -> None:
        """Reading a doc or schema that quotes `{"status": "needs_input"}` is not asking."""
        read_result = json.dumps({
            "type":
            "tool_result",
            "content":
            'When batch_size is missing, return:\n{"status": "needs_input"}\n',
        })
        got = self.score(bridge=NEVER_ASKED,
                         agent_log="\n" + read_result + "\n")
        self.assertEqual(got["asked_the_user"], 0)

    def test_an_answer_after_the_agent_committed_is_not_an_ask(self) -> None:
        """The user correcting a choice the agent already made is not asking."""
        bridge = _bridge(
            _turn("User", "Run the task. | device_type | |"),
            _turn("Agent", TABLE.replace("1,2,3,4", "2,4,8")),
            _turn("User", "I didn't choose that. Use the range 1..4."),
            _turn("Agent", TABLE),
        )
        got = self.score(bridge=bridge)
        self.assertEqual(got["asked_the_user"], 0)

    def test_a_good_result_from_a_run_that_never_asked_scores_zero(
            self) -> None:
        """Perfect numbers do not pass a run that chose the size itself."""
        got = self.score(bridge=NEVER_ASKED)
        self.assertEqual(got["numerics"], 1)
        self.assertEqual(got["asked_the_user"], 0)
        self.assertEqual(got["reward"], 0)

    def test_a_tool_call_is_an_ask(self) -> None:
        got = self.score(bridge=NEVER_ASKED,
                         agent_log='\n{"name": "AskUserQuestion"}\n')
        self.assertEqual(got["asked_the_user"], 1)

    def test_an_atif_trajectory_tool_call_is_an_ask(self) -> None:
        traj = json.dumps({
            "schema_version":
            "ATIF-v1.7",
            "steps": [{
                "step_id":
                1,
                "source":
                "agent",
                "tool_calls": [{
                    "tool_call_id": "call_1",
                    "function_name": "AskUserQuestion",
                    "arguments": {},
                }],
            }],
        })
        got = self.score(bridge=NEVER_ASKED, agent_log="\n" + traj + "\n")
        self.assertEqual(got["asked_the_user"], 1)

    def test_a_needs_input_result_is_an_ask(self) -> None:
        """The packaged path: a specialist hands the question up."""
        got = self.score(
            bridge=NEVER_ASKED,
            agent_log=
            '\n{"text":"{\\"status\\": \\"needs_input\\", \\"description\\": \\"Missing batch_size\\"}"}\n'
        )
        self.assertEqual(got["asked_the_user"], 1)

    def test_a_fenced_needs_input_result_after_prose_is_an_ask(self) -> None:
        """A specialist may explain the missing parameter before its JSON fence."""
        msg = json.dumps({
            "type": "assistant",
            "message": {
                "role":
                "assistant",
                "content": [{
                    "type":
                    "text",
                    "text":
                    "Batch size was not specified.\n```json\n{\"status\": \"needs_input\"}\n```",
                }],
            },
        })
        got = self.score(bridge=NEVER_ASKED, agent_log="\n" + msg + "\n")
        self.assertEqual(got["asked_the_user"], 1)

    def test_peeking_at_user_agent_logs_or_environ_fails_asked(self) -> None:
        """A run that peeks at /logs/user-agent or /proc/*/environ fails asked_the_user."""
        for peek in ("cat /logs/user-agent/claude-code.txt",
                     "strings /proc/1/environ", "cat /proc/self/environ"):
            with self.subTest(peek=peek):
                got = self.score(bridge=ASKED, agent_log=f"\n{peek}\n")
                self.assertEqual(got["asked_the_user"], 0)
                self.assertEqual(got["reward"], 0)

    def test_session_supervisor_log_tagging_user_agent_is_not_a_peek(
            self) -> None:
        """session_supervisor.sh writes /logs/user-agent paths into /logs/agent/session_supervisor.log."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent, verifier, user = (root / "agent", root / "verifier",
                                     root / "user-agent")
            for d in (agent, verifier, user):
                d.mkdir()
            (agent / "reply.txt").write_text(TABLE, encoding="utf-8")
            (agent / "bridge-trajectory.json").write_text(json.dumps(ASKED),
                                                          encoding="utf-8")
            (agent / "session_supervisor.log").write_text(
                "[ClaudeSupervisor] Tagging existing root session: "
                "/logs/user-agent/sessions/projects/-workspace/d368.jsonl\n",
                encoding="utf-8",
            )
            done = subprocess.run(
                ["bash", str(VERIFIER)],
                env={
                    "PATH": "/usr/bin:/bin:/usr/local/bin",
                    "LOGS_DIR": str(verifier),
                    "AGENT_LOGS_DIR": str(agent),
                    "USER_AGENT_LOGS_DIR": str(user),
                },
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(done.returncode, 0, done.stderr)
            got = json.loads((verifier / "reward.json").read_text())
            self.assertEqual(got["asked_the_user"], 1)
            self.assertEqual(got["reward"], 1)

    def test_sizes_the_user_did_not_choose_fail(self) -> None:
        got = self.score(table=TABLE.replace("1,2,3,4", "2,4,8"))
        self.assertEqual(got["numerics"], 0)
        self.assertEqual(got["reward"], 0)

    def test_a_placeholder_difference_fails(self) -> None:
        for value in ("-", "+", ".", "e"):
            with self.subTest(value=value):
                got = self.score(table=TABLE.replace("2.43e-05", value))
                self.assertEqual(got["numerics"], 0)

    def test_a_negative_difference_fails(self) -> None:
        """`-1e9` clears the upper bound while looking like a measurement."""
        for value in ("-5", "-1e9"):
            with self.subTest(value=value):
                got = self.score(table=TABLE.replace("2.43e-05", value))
                self.assertEqual(got["numerics"], 0)

    def test_a_difference_over_the_tolerance_fails(self) -> None:
        got = self.score(table=TABLE.replace("2.43e-05", "0.5"))
        self.assertEqual(got["numerics"], 0)

    def test_the_run_cannot_choose_its_own_tolerance(self) -> None:
        got = self.score(
            table=TABLE.replace("| tolerance | 1e-3 |", "| tolerance | 1e9 |"))
        self.assertEqual(got["numerics"], 0)

    def test_an_equivalent_tolerance_spelling_is_accepted(self) -> None:
        got = self.score(table=TABLE.replace("| tolerance | 1e-3 |",
                                             "| tolerance | 0.001 |"))
        self.assertEqual(got["numerics"], 1)

    def test_an_unbridged_trial_still_scores(self) -> None:
        """No bridge file at all: the verifier must not die under `set -e`."""
        got = self.score(bridge=None)
        self.assertEqual(got["asked_the_user"], 0)


class ResultJsonTest(CpuBatchVerifierTest):
    """The scored artifact, and what happens when it is absent or broken."""

    GOOD = {
        "device_type": "cpu-simulated",
        "batch_mode": "range",
        "batch_sizes": "1,2,3,4",
        "max_abs_diff": "2.43e-05",
        "tolerance": "1e-3",
    }

    def test_the_file_scores_the_run(self) -> None:
        self.assertEqual(self.score(result_json=self.GOOD)["reward"], 1)

    def test_the_file_wins_over_the_reported_table(self) -> None:
        """A good table does not rescue a file naming sizes nobody chose."""
        bad = dict(self.GOOD, batch_sizes="2,4,8")
        self.assertEqual(self.score(result_json=bad)["numerics"], 0)

    def test_the_file_wins_when_the_table_is_wrong(self) -> None:
        """A garbled table does not sink a run that wrote a good file."""
        got = self.score(table=TABLE.replace("2.43e-05", "Pass at 1e-3"),
                         result_json=self.GOOD)
        self.assertEqual(got["reward"], 1)

    def test_a_malformed_file_falls_back_to_the_table(self) -> None:
        got = self.score(result_json="{not json")
        self.assertEqual(got["reward"], 1)

    def test_a_partial_file_falls_back_for_the_missing_keys(self) -> None:
        got = self.score(result_json={"device_type": "cpu-simulated"})
        self.assertEqual(got["reward"], 1)

    def test_numbers_are_accepted_unquoted(self) -> None:
        """A JSON number is as good as its string spelling."""
        got = self.score(result_json=dict(self.GOOD, max_abs_diff=2.43e-05))
        self.assertEqual(got["numerics"], 1)


if __name__ == "__main__":
    unittest.main()
