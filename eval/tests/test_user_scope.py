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
"""The user-scope agents, and the dataset that scores what they install."""

import importlib.util
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from eval.utils import USER_SCOPE_ANSWERS, USER_SCOPE_DIST, find_repo_root

REPO = find_repo_root()
PACKAGE = REPO / "package"
DATASET = REPO / "eval" / "datasets" / "user-scope-install"
CHECK_MEMORY = (DATASET / "memory-loaded" / "tests" / "memory_loaded" /
                "check_memory.py")


def load_check_memory():
    spec = importlib.util.spec_from_file_location("check_memory", CHECK_MEMORY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(importlib.util.find_spec("harbor"), "needs harbor")
class TestUserScopeAgents(unittest.TestCase):
    """Each agent installs before the harness starts, then bridges whatever
    config directory Harbor relocated."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _command(self, cls_name: str) -> str:
        from eval.agents import user_scope
        agent = getattr(user_scope, cls_name)(logs_dir=Path(self.tmp.name),
                                              model_name="x/y")
        return agent._build_register_skills_command()

    def test_every_agent_runs_the_user_scope_install(self) -> None:
        for name in ("UserScopeClaudeCode", "UserScopeCodex",
                     "UserScopeCodexHome", "UserScopeAntigravityCli"):
            with self.subTest(agent=name):
                command = self._command(name)
                self.assertIn(
                    f"{USER_SCOPE_DIST}/framework/installer.py install"
                    f" {USER_SCOPE_DIST} --scope user", command)
                self.assertIn(f"--answers {USER_SCOPE_ANSWERS}", command)
                # The host packages docs/install.md has a user install, or
                # the session stops to ask for them before doing anything.
                self.assertLess(
                    command.index(
                        f"-r {USER_SCOPE_DIST}/requirements-client.txt"),
                    command.index("installer.py install"))

    def test_claude_code_sees_its_default_config_dir(self) -> None:
        command = self._command("UserScopeClaudeCode")
        install = command.index("installer.py")
        self.assertGreater(command.index('"$CLAUDE_CONFIG_DIR/"'), install)

    def test_codex_installs_into_the_default_home_then_bridges(self) -> None:
        command = self._command("UserScopeCodex")
        self.assertTrue(command.startswith("env -u CODEX_HOME "))
        self.assertIn('"$CODEX_HOME/AGENTS.md"', command)
        self.assertIn('>> "$CODEX_HOME/config.toml"', command)

    def test_codex_home_installs_into_codex_home_itself(self) -> None:
        """The install runs with Harbor's CODEX_HOME, which is under $HOME,
        and nothing is carried across after it."""
        from eval.agents import user_scope
        home = user_scope.UserScopeCodexHome._REMOTE_CODEX_HOME
        self.assertTrue(home.is_relative_to("/root"))
        command = self._command("UserScopeCodexHome")
        self.assertNotIn("env -u CODEX_HOME", command)
        self.assertNotIn("cp ", command)
        self.assertIn("user-scope-install.txt", command)
        self.assertIn('sha256sum "$CODEX_HOME/AGENTS.md"', command)

    def test_install_record_keeps_no_block_content(self) -> None:
        """The record names the block by its markers, never by what it says:
        the session can read the record, and the answers are in the block."""
        from eval.agents import user_scope
        tmp = Path(self.tmp.name)
        memory = tmp / "AGENTS.md"
        memory.write_text("<!-- pkg:pkg:kb:begin -->\nthe secret answer\n"
                          "<!-- pkg:pkg:kb:end -->\n")
        record = tmp / "record.txt"
        subprocess.run([
            "bash", "-c",
            user_scope.install_record(str(record), "echo v1", str(memory),
                                      str(tmp))
        ],
                       check=True)
        text = record.read_text()
        self.assertIn("v1", text)
        self.assertIn("1:<!-- pkg:pkg:kb:begin -->", text)
        self.assertIn("3:<!-- pkg:pkg:kb:end -->", text)
        self.assertNotIn("the secret answer", text)

    def test_antigravity_bridges_nothing(self) -> None:
        """It reads ~/.gemini to record what the session will find, and
        copies nothing anywhere."""
        command = self._command("UserScopeAntigravityCli")
        self.assertNotIn("cp ", command)
        self.assertIn("user-scope-install.txt", command)

    def test_configs_name_agents_that_exist(self) -> None:
        from eval.agents import user_scope
        configs = sorted(
            (REPO / "eval" / "configs").glob("*/user-scope*.yaml"))
        # One per harness, and Codex's CODEX_HOME variant; an empty glob
        # would pass without checking any.
        self.assertEqual([
            "agy/user-scope.yaml", "claude-code/user-scope.yaml",
            "codex/user-scope-codex-home.yaml", "codex/user-scope.yaml"
        ], [f"{path.parent.name}/{path.name}" for path in configs])
        for path in configs:
            with self.subTest(config=str(path)):
                agent = yaml.safe_load(path.read_text())["agents"][0]
                module, cls = agent["import_path"].split(":")
                self.assertEqual(module, user_scope.__name__)
                self.assertTrue(hasattr(user_scope, cls))
                self.assertNotIn("name", agent)  # a name wins over the path


class TestMemoryLoadedExpectations(unittest.TestCase):
    """The memory-loaded task scores two facts only the memory block states;
    if either drifts out of the block, or into what every session sees
    anyway, the task stops measuring whether the block was loaded."""

    def setUp(self) -> None:
        self.check = load_check_memory()

    def _descriptions(self) -> str:
        texts = []
        for path in [
                *PACKAGE.rglob("SKILL.md"), *(PACKAGE / "agents").glob("*.md")
        ]:
            match = re.match(r"---\n(.*?)\n---", path.read_text(), re.S)
            if match:
                texts.append(
                    str((yaml.safe_load(match.group(1))
                         or {}).get("description", "")))
        return "\n".join(texts)

    def test_every_answer_is_in_the_memory_block(self) -> None:
        manifest = json.loads((PACKAGE / "package.release.json").read_text())
        blocks = "\n".join((PACKAGE / entry["content"]).read_text()
                           for entry in manifest["workspace_memory"])
        for wanted in self.check.EXPECT.values():
            self.assertIn(wanted, blocks)

    def test_no_answer_is_in_a_description(self) -> None:
        descriptions = self._descriptions()
        for wanted in self.check.EXPECT.values():
            self.assertNotIn(wanted, descriptions)

    def test_an_answer_naming_both_passes(self) -> None:
        self.assertEqual([],
                         self.check.check_answer({
                             "kernel_index": "kernel-optimization-index.md",
                             "request_field": "`entry_id`",
                         }))

    def test_unknown_fails(self) -> None:
        self.assertEqual(
            2,
            len(
                self.check.check_answer({
                    "kernel_index": "unknown",
                    "request_field": "unknown"
                })))

    def _reply(self, message) -> dict:
        return {
            "steps": [
                {
                    "source": "user",
                    "message": "entry_id kernel-optimization-index"
                },
                {
                    "source": "agent",
                    "message": message
                },
            ]
        }

    def test_a_reply_without_the_file_is_scored(self) -> None:
        """agy answered both in its reply and wrote no file (PR #619 CI)."""
        reply = self._reply("1. `kernel-optimization-index.md`\n"
                            "2. `entry_id`")
        with mock.patch.object(self.check, "ANSWER",
                               Path(tempfile.mkdtemp()) / "none.json"):
            result = self.check.found_answer(reply)
        self.assertEqual([], self.check.check_answer(result))

    def test_the_question_is_not_the_answer(self) -> None:
        """Only the agent's reply counts, never the prompt's text."""
        reply = self._reply([{"type": "text", "text": "unknown"}])
        with mock.patch.object(self.check, "ANSWER",
                               Path(tempfile.mkdtemp()) / "none.json"):
            result = self.check.found_answer(reply)
        self.assertEqual(2, len(self.check.check_answer(result)))

    def test_reading_the_memory_file_fails(self) -> None:
        trajectory = {
            "steps": [{
                "tool_calls": [{
                    "function_name": "Read",
                    "arguments": {
                        "file_path": "/root/.claude/CLAUDE.md"
                    },
                }]
            }]
        }
        why = self.check.check_looked(trajectory)
        self.assertEqual(1, len(why))
        self.assertTrue(why[0].startswith("Read touched CLAUDE.md: "))

    def _viewed(self, path: str) -> dict:
        return {
            "steps": [{
                "tool_calls": [{
                    "function_name": "view_file",
                    "arguments": {
                        "AbsolutePath": path
                    },
                }]
            }]
        }

    def test_reading_the_installed_package_fails(self) -> None:
        self.assertEqual(
            1,
            len(
                self.check.check_looked(
                    self._viewed("/root/.tpu-agent-suite/packages/"
                                 "tpu-agent-suite@0.5.0/data/x.md"))))

    def test_a_harness_reading_its_own_files_passes(self) -> None:
        """agy views its brain artifacts under ~/.gemini unprompted (#619)."""
        self.assertEqual([],
                         self.check.check_looked(
                             self._viewed("/root/.gemini/antigravity-cli/"
                                          "brain/abc/task.md")))

    def _called(self, name: str, **arguments) -> list[str]:
        return self.check.check_looked({
            "steps": [{
                "tool_calls": [{
                    "function_name": name,
                    "arguments": arguments
                }]
            }]
        })

    def test_a_read_that_writes_the_answer_still_fails(self) -> None:
        """One call that reads the block and writes answer.json (#619)."""
        self.assertTrue(
            self._called("run_command",
                         CommandLine="grep entry_id ~/.gemini/AGENTS.md"
                         " > /logs/artifacts/answer.json"))

    def test_reads_that_do_not_name_the_file_fail(self) -> None:
        """Globs and searches reach the block without naming it (#619)."""
        for command in ("cat ~/.gemini/A*",
                        "find ~ -name '*.md' -exec cat {} +",
                        "grep -rn kernel ~/", "rg entry ~"):
            with self.subTest(command=command):
                self.assertTrue(self._called("Bash", command=command))
        self.assertTrue(self._called("Grep", pattern="kernel", path="/root"))

    def test_writing_the_answer_alone_passes(self) -> None:
        self.assertEqual([],
                         self._called(
                             "Bash",
                             command="mkdir -p /logs/artifacts && echo"
                             " '{\"request_field\": \"entry_id\"}'"
                             " > /logs/artifacts/answer.json"))

    def test_stating_the_answer_is_not_looking_for_it(self) -> None:
        """agy sends its answer as a message, naming both (#619)."""
        trajectory = {
            "steps": [{
                "tool_calls": [{
                    "function_name": "send_message",
                    "arguments": {
                        "Message":
                        '{"kernel_index": '
                        '"kernel-optimization-index.md", '
                        '"request_field": "entry_id"}',
                    },
                }]
            }]
        }
        self.assertEqual([], self.check.check_looked(trajectory))

    def test_no_answer_is_installed_into_a_harness_directory(self) -> None:
        """What lets check_looked leave ~/.claude, ~/.codex and ~/.gemini
        alone: what is installed there, but the memory file, is compiled
        from these sources, and none of them carries an answer."""
        for root in ("agents", "skills"):
            for path in (PACKAGE / root).rglob("*"):
                if not path.is_file():
                    continue
                text = path.read_text(errors="ignore")
                for wanted in self.check.EXPECT.values():
                    with self.subTest(path=str(path), wanted=wanted):
                        self.assertNotIn(wanted, text)

    def test_writing_the_answer_is_allowed(self) -> None:
        trajectory = {
            "steps": [{
                "tool_calls": [{
                    "function_name": "Write",
                    "arguments": {
                        "file_path": "/logs/artifacts/answer.json",
                        "content": '{"request_field": "entry_id"}',
                    },
                }]
            }]
        }
        self.assertEqual([], self.check.check_looked(trajectory))


if __name__ == "__main__":
    unittest.main()


class TestDelegationCheck(unittest.TestCase):
    """agent-delegation reads the trajectory, or Claude Code's stream log
    when Harbor wrote no trajectory for a run with several sessions."""

    CHECK = (DATASET / "agent-delegation" / "tests" /
             "delegated_to_knowledge_base" / "check_delegation.py")

    def setUp(self) -> None:
        spec = importlib.util.spec_from_file_location("check_delegation",
                                                      self.CHECK)
        self.check = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.check)
        self.tmp = Path(tempfile.mkdtemp())

    def _stream(self, *tool_uses: dict) -> Path:
        path = self.tmp / "claude-code.txt"
        lines = ["not json", json.dumps({"type": "system"})]
        lines += [
            json.dumps({
                "type": "assistant",
                "message": {
                    "content": [{
                        "type": "tool_use",
                        **use
                    }]
                },
            }) for use in tool_uses
        ]
        path.write_text("\n".join(lines))
        return path

    def test_the_stream_log_stands_in_for_a_missing_trajectory(self) -> None:
        """The Agent call PR #619's Claude Code run made, with no trajectory."""
        stream = self._stream({
            "name": "Agent",
            "input": {
                "subagent_type": "tpu-knowledge-base",
                "prompt": '{"query": "What is the HBM capacity of v6e?"}',
            },
        })
        calls, source = self.check.found_calls(self.tmp / "none.json", stream)
        self.assertEqual(stream, source)
        self.assertEqual("Agent",
                         self.check.delegated(calls, ["tpu-knowledge-base"]))

    def test_a_trajectory_is_read_first(self) -> None:
        trajectory = self.tmp / "trajectory.json"
        trajectory.write_text(
            json.dumps({
                "steps": [{
                    "tool_calls": [{
                        "function_name": "invoke_subagent",
                        "arguments": {
                            "name": "tpu_knowledge_base"
                        },
                    }]
                }]
            }))
        calls, source = self.check.found_calls(trajectory, self._stream())
        self.assertEqual(trajectory, source)
        self.assertEqual("invoke_subagent",
                         self.check.delegated(calls, ["tpu-knowledge-base"]))

    def test_other_tools_and_other_agents_do_not_count(self) -> None:
        stream = self._stream(
            {
                "name": "Bash",
                "input": {
                    "command": "echo tpu-knowledge-base"
                }
            },
            {
                "name": "Agent",
                "input": {
                    "subagent_type": "tpu-model-onboarding"
                }
            },
        )
        calls, _ = self.check.found_calls(self.tmp / "none.json", stream)
        self.assertIsNone(self.check.delegated(calls, ["tpu-knowledge-base"]))


class TestAnswerCheck(unittest.TestCase):
    """agent-delegation scores the answer file, or the agent's last message
    when it wrote none: agy's persona has no tool that writes a file."""

    CHECK = (DATASET / "agent-delegation" / "tests" / "answer_v6e_hbm" /
             "check_answer.py")

    def setUp(self) -> None:
        spec = importlib.util.spec_from_file_location("check_answer",
                                                      self.CHECK)
        self.check = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.check)
        self.tmp = Path(tempfile.mkdtemp())

    def _found(self, message) -> dict | None:
        trajectory = self.tmp / "trajectory.json"
        trajectory.write_text(
            json.dumps({
                "steps": [{
                    "source": "user",
                    "message": "What is the HBM capacity per chip?"
                }, {
                    "source": "agent",
                    "message": message
                }]
            }))
        with mock.patch.object(self.check, "ANSWER",
                               self.tmp / "none.json"), mock.patch.object(
                                   self.check, "TRAJECTORIES", (trajectory, )):
            return self.check.found_result(self.tmp)

    def test_a_reply_without_the_file_is_scored(self) -> None:
        """agy's reply on PR #619, which wrote no answer.json."""
        result = self._found('```json\n{"answer": "32 GB", "reference": '
                             '"https://cloud.google.com/tpu/docs/v6e"}\n```')
        self.assertEqual((True, []), self.check.check(result,
                                                      self.check.EXPECT))

    def test_either_host_of_the_v6e_page_counts(self) -> None:
        for host in ("docs.cloud.google.com", "cloud.google.com"):
            with self.subTest(host=host):
                ok, _ = self.check.check(
                    {
                        "answer": "32 GB",
                        "reference": f"https://{host}/tpu/docs/v6e"
                    }, self.check.EXPECT)
                self.assertTrue(ok)

    def test_a_wrong_reply_fails(self) -> None:
        result = self._found("I could not find it.")
        ok, _ = self.check.check(result, self.check.EXPECT)
        self.assertFalse(ok)
