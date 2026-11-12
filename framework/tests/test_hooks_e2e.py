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
"""End-to-end CLI integration tests for lifecycle hooks integrating real harnesses."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Callable

from framework.jsonio import read_jsonl
from framework.paths import TPU_AGENT_SUITE_HOME_SPELLING
from framework.utils import (RUN_ID_PATTERN, SemVer, extract_all_json_objects,
                             validate_instance, version_satisfies)

# How each harness CLI runs one prompt: the words before it, the words after
# it, and the hook config the installer writes into the workspace.
HARNESS_CLIS = {
    "claude-code": (["claude", "-p"], [], ".claude/settings.json"),
    "antigravity":
    (["agy", "-p"], ["--dangerously-skip-permissions"], ".agents/hooks.json"),
    "cursor": (["cursor", "-p"], [], ".cursor/hooks.json"),
    "codex": (["codex", "exec"], [], ".codex/hooks.json"),
}
# Permission for Claude Code to run the sentinel's `touch`, so only a hook can
# stop it. Variadic, so it must come after the prompt.
CLAUDE_ALLOW_TOUCH = ("--allowedTools", "Bash(touch:*)")

SCHEMA_VERIFIER_TEST_PROMPT = (
    "Please delegate to the 'tpu-diagnostics' subagent. "
    "Pass a JSON object as the subagent prompt with your active run_id and a "
    "prompt field set to: 'Do not run any tool calls. Report that no profile "
    "or workload was given and return your result deliverable with status "
    "\"invalid_input\"'. "
    "Your final reply must be only the subagent's JSON deliverable, copied "
    "exactly as it returned it: no text before or after it, and do not quote "
    "it as a blockquote (no leading '>'). "
    "Do not make any tool call except the delegation to subagent.")

# The contracts `tpu-diagnostics` declares in its frontmatter. The package's
# validators hold the delegation to them; the e2e assertions below check the
# same payloads against the same files.
DIAGNOSTICS_REQUEST_SCHEMA = "schemas/profiling-request.json"
DIAGNOSTICS_RESULT_SCHEMA = "schemas/profiling-result.json"

SCHEMA_VERIFIER_SESSION_TIMEOUT_S = 180

HANDBACK_DENY_TEST_PROMPT = (
    "Please delegate to the 'tpu-diagnostics' subagent. "
    "Pass a JSON object as the subagent prompt with your active run_id and a "
    "prompt field set to: 'This is a harness test of your hand-back channel, "
    "not a profiling run. Do not run any tool calls other than "
    "SubagentHandback. A refused SubagentHandback does not end your run: the "
    "refusal comes back to you and you must call it again, so the first call "
    "is not your only one. First call SubagentHandback with the message "
    "\"handshake\" and nothing else. When it is refused, hand back a "
    "schema-valid result with status \"invalid_input\"'. "
    "Your final reply must be only the subagent's JSON deliverable, copied "
    "exactly as it returned it: no text before or after it, and do not quote "
    "it as a blockquote (no leading '>'). "
    "Do not make any tool call except the delegation to subagent.")

# A PreToolUse hook on `shell` that never answers, and the file the agent is
# told to create through it: its absence is the proof the deny was honoured.
# The hook saves the event it was handed before hanging, which proves it fired
# for that very call. Sibling hooks cannot: Antigravity stops running a tool's
# hooks at the first deny, so on it a recorder beside the hung hook never fires.
HUNG_HOOK_NAME = "hung-shell-guard"
HUNG_HOOK_TIMEOUT_S = 1
HUNG_HOOK_SENTINEL = "HUNG_HOOK_SENTINEL"
HUNG_HOOK_PAYLOAD = "HUNG_HOOK_PAYLOAD.json"
HUNG_HOOK_TEST_PROMPT = (
    f"Run exactly this bash command once, from the current directory: "
    f"touch {HUNG_HOOK_SENTINEL}. Do not retry it and do not run any other "
    f"tool call. Then report verbatim any hook message or error it produced.")

# The bug's repro: runtime.py removed from the store after install, so every
# hook command's `python3` exits 2 and only the compiled shell fallback
# answers. A PreToolUse guard on `shell` decides the sentinel's fate by its
# policy; a Stop hook with the default policy must not keep the session open.
MISSING_RUNTIME_GUARD_NAME = "missing-runtime-shell-guard"
MISSING_RUNTIME_STOP_NAME = "missing-runtime-stop"
MISSING_RUNTIME_SENTINEL = "MISSING_RUNTIME_SENTINEL"
MISSING_RUNTIME_SESSION_TIMEOUT_S = 180
MISSING_RUNTIME_TEST_PROMPT = (
    f"Run exactly this bash command once, from the current directory: "
    f"touch {MISSING_RUNTIME_SENTINEL}. Do not retry it and do not run any "
    f"other tool call. Then reply with one line and stop.")

# The delegation assertions read what the hooks were handed: the request on
# PreToolUse, and every SubagentStop fire with the result it carried. The
# shipped package records none of that, so the test package copy carries a
# recorder of its own: one script, registered on both events, that appends
# each payload it receives to a log in the workspace. It is appended after
# the package's validators so it sees every fire they settle, refusals too.
EVENT_RECORDER_SCRIPT = "scripts/hooks/record_hook_event.py"
EVENT_RECORDER_LOG = ".tpu-agent-suite/logs/hook_events.jsonl"
EVENT_RECORDER_EVENTS = ("PreToolUse", "SubagentStop")
EVENT_RECORDER_SOURCE = f'''#!/usr/bin/env python3
"""Test-only hook: appends the event payload it was handed to the workspace log."""

from __future__ import annotations

import datetime
import json
import pathlib
import sys


def main() -> int:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        payload = {{}}
    workspace = (payload.get("context") or {{}}).get("workspace") or "."
    log_file = pathlib.Path(workspace) / {EVENT_RECORDER_LOG!r}
    log_file.parent.mkdir(parents=True, exist_ok=True)
    record = {{
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        **payload,
    }}
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\\n")
    json.dump({{"decision": "allow"}}, sys.stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

# Claude Code delivers a subagent's result by one of two roads: outside auto
# mode, the subagent's final message (native SubagentStop); in auto mode,
# from this release on, a call to the injected `SubagentHandback` tool. The
# runtime tells the roads apart by payload and never checks a version; only
# these tests do, to know whether the second road can be exercised at all.
CLAUDE_HANDBACK_TOOL = "SubagentHandback"
CLAUDE_HANDBACK_MIN_VERSION = "2.1.271"


def _claude_version() -> SemVer | None:
    """The installed Claude Code version, or None when it cannot be read."""
    try:
        res = subprocess.run(["claude", "--version"],
                             capture_output=True,
                             text=True,
                             timeout=30)
        return SemVer.parse(res.stdout)
    except Exception:
        return None


def _claude_has_handback() -> bool:
    version = _claude_version()
    return version is not None and version_satisfies(
        version, f">={CLAUDE_HANDBACK_MIN_VERSION}")


def _needs_cli(harness: str):
    """Skip unless the harness's CLI binary is on PATH."""
    binary = HARNESS_CLIS[harness][0][0]
    return unittest.skipIf(not shutil.which(binary),
                           f"{binary} CLI binary not found in PATH")


@unittest.skipUnless(
    os.environ.get("RUN_HARNESS_CLI_E2E"),
    "Set RUN_HARNESS_CLI_E2E=1 to run live harness CLI end-to-end tests",
)
class TestHooksCLIEndToEnd(unittest.TestCase):
    """End-to-end tests imitating user CLI compilation, installation, and harness execution.

    Every install lands in the test's temp dir and the harness reads it from
    the workspace. Google harnesses merge ~/.gemini/config into the session
    as well: run these with no global hooks or agents there, or the session
    is not the one under test.
    """

    def setUp(self):
        self._start_time = time.monotonic()
        super().setUp()
        self.repo_root = Path(__file__).resolve().parents[2]
        self.package_dir = self.repo_root / "package"
        self.framework_dir = self.repo_root / "framework"

        self.keep_output = bool(os.environ.get("KEEP_TEST_OUTPUT"))
        self.keep_install = bool(os.environ.get("KEEP_TEST_INSTALL"))

        if self.keep_output or self.keep_install:
            base_tmp = self.repo_root / "tmp"
            base_tmp.mkdir(parents=True, exist_ok=True)
            self.temp_dir = base_tmp / f"e2e_{self._testMethodName}"
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.temp_dir = Path(tempfile.mkdtemp())
        self.central_home = self.temp_dir / "central_tpu_agent_suite"
        self.central_home.mkdir(parents=True, exist_ok=True)

        old_tpu_agent_suite_home = os.environ.get("TPU_AGENT_SUITE_HOME")
        os.environ["TPU_AGENT_SUITE_HOME"] = str(self.central_home)

        def _restore_env():
            if old_tpu_agent_suite_home is not None:
                os.environ["TPU_AGENT_SUITE_HOME"] = old_tpu_agent_suite_home
            else:
                os.environ.pop("TPU_AGENT_SUITE_HOME", None)

        self.addCleanup(_restore_env)

    def tearDown(self):
        elapsed = time.monotonic() - getattr(self, "_start_time",
                                             time.monotonic())
        print(f"\n[{self._testMethodName}] took {elapsed:.2f}s")
        if not self.keep_output and not self.keep_install:
            shutil.rmtree(self.temp_dir, ignore_errors=True)
        else:
            if not self.keep_install:
                for p in self.temp_dir.glob("dist_*"):
                    shutil.rmtree(p, ignore_errors=True)
                for p in self.temp_dir.glob("workspace_*"):
                    shutil.rmtree(p, ignore_errors=True)
            if not self.keep_output:
                for log_file in self.temp_dir.glob("*.log"):
                    log_file.unlink(missing_ok=True)
                shutil.rmtree(self.central_home / "runs", ignore_errors=True)
            print(
                f"[TestHooksCLIEndToEnd] Preserved artifacts in: {self.temp_dir}"
            )

        super().tearDown()

    def _setup_and_run_harness_cli(
        self,
        harness: str,
        prompt: str,
        name: str,
        extra_args: tuple[str, ...] = (),
        timeout: int = 90,
        package_dir: Path | None = None,
        before_run: Callable[[Path], None] | None = None,
        expect_run: bool = True,
    ) -> tuple[subprocess.CompletedProcess[str], str, Path]:
        """Compile, install, and execute harness CLI end-to-end, asserting lifecycle initialization.

        Runs `prompt` through the harness's CLI (`HARNESS_CLIS`), followed by
        `extra_args`; `name` names the dist, workspace and logs.
        `before_run` is called with the workspace after the install is
        complete and patched, just before the harness is launched: for a
        test that breaks the install on purpose. `expect_run=False` skips the
        run-namespace assertions, for a test whose dispatcher cannot run.
        Returns the CLI result, the run id ("" without a run) and workspace.
        """
        before, after, hook_rel_path = HARNESS_CLIS[harness]
        cli_cmd = [*before, prompt, *after, *extra_args]
        log_prefix = name
        dist_dir = self.temp_dir / f"dist_{name}"
        workspace = self.temp_dir / f"workspace_{name}"
        workspace.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init"],
                       cwd=str(workspace),
                       capture_output=True)

        # 1. Compile package via compiler CLI
        pkg_dir = package_dir if package_dir is not None else self.package_dir
        compile_cmd = [
            sys.executable,
            str(self.framework_dir / "compiler.py"),
            "translate",
            str(pkg_dir),
            "--harness",
            harness,
            "--output",
            str(dist_dir),
        ]
        compile_res = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            cwd=str(self.repo_root),
        )
        self.assertEqual(
            compile_res.returncode,
            0,
            f"Compiler CLI failed with code {compile_res.returncode}:\n{compile_res.stderr}",
        )
        self.assertTrue((dist_dir / "tpu-agent-suite-build.json").is_file())

        # 2. Install package into workspace via installer CLI
        install_cmd = [
            sys.executable,
            str(self.framework_dir / "installer.py"),
            "install",
            str(dist_dir),
            "--destination",
            str(workspace),
            "--non-interactive",
            "--skip-harness-check",
            "--relocated-store",
        ]
        install_res = subprocess.run(
            install_cmd,
            capture_output=True,
            text=True,
            cwd=str(self.repo_root),
        )
        self.assertEqual(
            install_res.returncode,
            0,
            f"Installer CLI failed with code {install_res.returncode}:\n{install_res.stderr}",
        )
        hook_config_file = workspace / hook_rel_path
        self.assertTrue(
            hook_config_file.is_file(),
            f"Expected hook configuration file at {hook_config_file}",
        )

        # The compiler home-anchors every central-store path it emits -- hook
        # commands, sandbox allowlists -- because a distribution is built once
        # and installed on any machine, so the spelling has to be expanded by
        # the target's own shell. That expansion uses HOME and ignores
        # TPU_AGENT_SUITE_HOME, which means an unpatched config sends the
        # harness to the developer's real ~/.tpu-agent-suite: the hooks that
        # fire are whatever stale build happens to be installed there, not the
        # one this test just compiled. The suite would then report on code
        # that is not in the working tree. Repoint the configs at the store the
        # installer actually wrote to, which is the only way the isolation seam
        # covers the hooks as well as the data.
        installed_runtimes = sorted(
            (self.central_home / "packages").glob("*/framework/runtime.py"))
        self.assertTrue(
            installed_runtimes,
            f"Installer placed no runtime under {self.central_home}/packages; "
            "the isolation patch below would point the hooks at nothing.",
        )

        patched_configs: list[Path] = []
        for config_file in sorted(workspace.rglob("*")):
            if not config_file.is_file() or config_file.suffix not in (
                    ".json", ".toml"):
                continue
            text = config_file.read_text(encoding="utf-8", errors="ignore")
            if TPU_AGENT_SUITE_HOME_SPELLING not in text:
                continue
            config_file.write_text(text.replace(TPU_AGENT_SUITE_HOME_SPELLING,
                                                str(self.central_home)),
                                   encoding="utf-8")
            patched_configs.append(config_file)
        self.assertTrue(
            patched_configs,
            f"No installed config under {workspace} referenced "
            f"'{TPU_AGENT_SUITE_HOME_SPELLING}', so the harness cannot be reaching the "
            "isolated store; the run would silently test the real one.",
        )

        # Verify initial runs directory has no runs before launching the
        # session. Filtered by run-id shape for the same reason as the
        # post-session count below: `jobs/` is not a run.
        runs_dir = self.central_home / "runs"
        initial_runs = ([
            d.name for d in runs_dir.iterdir()
            if d.is_dir() and re.fullmatch(RUN_ID_PATTERN, d.name)
        ] if runs_dir.is_dir() else [])
        self.assertEqual(initial_runs, [])

        # 3. Run prompt end-to-end using real harness CLI
        env = dict(os.environ)
        env["TPU_AGENT_SUITE_HOME"] = str(self.central_home)

        if before_run is not None:
            before_run(workspace)

        try:
            cli_res = subprocess.run(
                cli_cmd,
                capture_output=True,
                text=True,
                cwd=str(workspace),
                env=env,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            if exc.stdout:
                (self.temp_dir / f"{log_prefix}_stdout.log").write_text(
                    exc.stdout if isinstance(exc.stdout, str) else
                    exc.stdout.decode("utf-8", errors="replace"),
                    encoding="utf-8",
                )
            if exc.stderr:
                (self.temp_dir / f"{log_prefix}_stderr.log").write_text(
                    exc.stderr if isinstance(exc.stderr, str) else
                    exc.stderr.decode("utf-8", errors="replace"),
                    encoding="utf-8",
                )
            self.fail(
                f"{harness} CLI execution timed out after {timeout}s: {exc}")

        # Save agent outputs into log files for inspection
        (self.temp_dir / f"{log_prefix}_stdout.log").write_text(
            cli_res.stdout, encoding="utf-8")
        (self.temp_dir / f"{log_prefix}_stderr.log").write_text(
            cli_res.stderr, encoding="utf-8")

        self.assertEqual(
            cli_res.returncode,
            0,
            f"{harness} CLI execution failed with code {cli_res.returncode}:\n{cli_res.stderr}",
        )

        if not expect_run:
            return cli_res, "", workspace

        # 4. Verify run namespace was created by session-run-init hook
        self.assertTrue(
            runs_dir.is_dir(),
            "Runs directory ~/.tpu-agent-suite/runs was not created")
        # The runs root holds bookkeeping siblings next to the run namespaces
        # -- `jobs/` for compute state, `current_run_id` for the active
        # pointer -- so identity has to come from the run-id shape rather than
        # from being a directory.
        allocated_runs = [
            d for d in runs_dir.iterdir()
            if d.is_dir() and re.fullmatch(RUN_ID_PATTERN, d.name)
        ]
        self.assertEqual(
            len(allocated_runs),
            1,
            f"Expected exactly 1 run namespace allocated, found: {allocated_runs}",
        )

        run_entry = allocated_runs[0]
        expected_run_id = run_entry.name
        run_json_path = run_entry / "run.json"
        self.assertTrue(run_json_path.is_file(),
                        f"Missing run.json in {run_entry}")

        manifest = json.loads(
            (self.package_dir /
             "package.release.json").read_text(encoding="utf-8"))
        run_data = json.loads(run_json_path.read_text(encoding="utf-8"))
        self.assertEqual(run_data.get("run_id"), expected_run_id)
        self.assertIn(manifest.get("name", "tpu-agent-suite"),
                      run_data.get("package", ""))
        self.assertTrue((run_entry / "artifacts").is_dir())

        return cli_res, expected_run_id, workspace

    def _copy_package(self, name: str, edit: Callable[[Path, dict],
                                                      None]) -> Path:
        """Copy the package to `name` in the temp dir and let `edit` change it.

        `edit` is handed the copy and its parsed manifest; the manifest is
        written back once it returns.
        """
        test_pkg = self.temp_dir / name
        if test_pkg.exists():
            shutil.rmtree(test_pkg, ignore_errors=True)
        shutil.copytree(self.package_dir, test_pkg)

        pkg_json_path = test_pkg / "package.release.json"
        pkg_manifest = json.loads(pkg_json_path.read_text(encoding="utf-8"))
        edit(test_pkg, pkg_manifest)
        pkg_json_path.write_text(json.dumps(pkg_manifest, indent=2),
                                 encoding="utf-8")
        return test_pkg

    def _create_schema_verifier_test_package(self) -> Path:
        """A package copy carrying the recorder the delegation assertions read.

        `tpu-diagnostics` ships its own contracts, `DIAGNOSTICS_REQUEST_SCHEMA`
        and `DIAGNOSTICS_RESULT_SCHEMA`, which the package's validators already
        enforce; the copy leaves the agent as it is and only adds the recorder.
        """
        return self._copy_package("pkg_schema_verifier",
                                  self._add_event_recorder)

    def _add_event_recorder(self, test_pkg: Path, pkg_manifest: dict) -> None:
        """Give a package copy the recorder the delegation assertions read.

        Writes the script into the copy and appends one hook per recorded
        event to `pkg_manifest`, which the caller writes back. Appending
        keeps the recorder after the package's validators, so it logs every
        fire they settle rather than only the ones they let through.
        """
        script = test_pkg / EVENT_RECORDER_SCRIPT
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text(EVENT_RECORDER_SOURCE, encoding="utf-8")
        hooks = pkg_manifest.setdefault("hooks", [])
        for event in EVENT_RECORDER_EVENTS:
            hooks.append({
                "name": f"record-{event.lower()}",
                "event": event,
                "action": f"script:{EVENT_RECORDER_SCRIPT}",
                "timeout": 10,
                "on_failure": "allow",
                "visibility": "external",
                "description": f"Test-only: records each {event} payload.",
                "action_level": {
                    "effect": "write",
                    "scope": "session",
                    "approval": "never",
                },
            })

    def _recorded_events(self, workspace: Path, event: str) -> list[dict]:
        """The payloads the recorder logged for `event`, in order."""
        return [
            e for e in read_jsonl(workspace / EVENT_RECORDER_LOG)
            if e.get("event") == event
        ]

    def _run_harness_cli_schema_verifier_e2e(self, harness: str, name: str):
        """Delegate to a subagent; the hooks enforce both contracts.

        The hooks hold request and result to their contracts, and the request
        carries the run_id SessionStart gave the caller -- its only source,
        since no tool call is allowed.
        """
        test_pkg = self._create_schema_verifier_test_package()
        cli_res, expected_run_id, workspace = self._setup_and_run_harness_cli(
            harness=harness,
            prompt=SCHEMA_VERIFIER_TEST_PROMPT,
            name=name,
            timeout=SCHEMA_VERIFIER_SESSION_TIMEOUT_S,
            package_dir=test_pkg,
        )

        # 1. Verify the incoming delegation request conforms to the request
        # contract
        pre_log_file = workspace / EVENT_RECORDER_LOG
        self.assertTrue(
            pre_log_file.is_file(),
            f"The recorder hook never wrote {pre_log_file}: no PreToolUse "
            f"fire reached it on {harness}.",
        )
        diagnostics_requests = [
            entry["args"]
            for entry in self._recorded_events(workspace, "PreToolUse")
            if isinstance(entry.get("args"), dict)
            and entry.get("subagent") == "tpu-diagnostics"
        ]
        self.assertTrue(
            diagnostics_requests,
            f"No 'tpu-diagnostics' delegation event found in {pre_log_file}.",
        )
        last_req = diagnostics_requests[-1]
        prompt_str = str(last_req.get("prompt", ""))
        json_cands = extract_all_json_objects(prompt_str)
        if json_cands:
            payload_to_verify = dict(json_cands[-1])
        else:
            payload_to_verify = {"prompt": prompt_str}
            if "run_id" in last_req:
                payload_to_verify["run_id"] = last_req["run_id"]

        diagnostics_request_schema = json.loads(
            (test_pkg /
             DIAGNOSTICS_REQUEST_SCHEMA).read_text(encoding="utf-8"))
        req_errors = validate_instance(payload_to_verify,
                                       diagnostics_request_schema)
        self.assertFalse(
            req_errors,
            f"Incoming delegation request payload violated "
            f"{DIAGNOSTICS_REQUEST_SCHEMA}: {req_errors}\nPayload: {payload_to_verify}",
        )
        self.assertEqual(
            payload_to_verify.get("run_id"), expected_run_id,
            f"{harness} agent was not aware of run_id {expected_run_id}.")

        # 2. Verify the delivered result conforms to the result contract: on
        # the last SubagentStop fire, the one the hooks let through, and in the
        # parent's output, which shows the harness actually delivered it.
        fires = self._diagnostics_stop_fires(workspace)
        self.assertTrue(
            fires, f"No SubagentStop hook fired for 'tpu-diagnostics' on "
            f"{harness}.")
        self.assertTrue(
            self._satisfies_diagnostics_result(fires[-1], test_pkg),
            f"The delivered result did not satisfy {DIAGNOSTICS_RESULT_SCHEMA}: "
            f"{fires[-1].get('output')!r}",
        )
        self._assert_delivered_in_stdout(cli_res.stdout, test_pkg)

        # 3. Claude Code can deliver by either of two roads; whichever it
        # took, the result must have been held to its contract by one alone.
        if harness == "claude-code":
            self._assert_checked_by_one_road(workspace)

    def _assert_checked_by_one_road(self,
                                    workspace: Path,
                                    require_handback: bool = False) -> dict:
        """Assert the delegated result was held to its contract by one road.

        Whichever road Claude Code picks -- the subagent's final message or a
        `SubagentHandback` call -- the SubagentStop hooks must fire for it,
        and never for both. A subagent given the tool can deliver only
        through it, so none of its native stops may be checked: not the one
        before the call (a draft the caller never sees -- the double check
        once observed in auto mode) nor the one after (a closing remark --
        the original failure, rejected on every retry).

        Returns the last fire, the one the hooks let through.
        """
        fires = self._diagnostics_stop_fires(workspace)
        self.assertTrue(fires,
                        "No SubagentStop hook fired for 'tpu-diagnostics'.")
        handbacks = [e for e in fires if (e.get("args") or {}).get("handback")]
        if require_handback:
            self.assertTrue(
                handbacks,
                f"Expected the subagent to deliver through "
                f"{CLAUDE_HANDBACK_TOOL}, but only native stops fired: {fires}",
            )
        if handbacks:
            self.assertEqual(
                handbacks,
                fires,
                "A subagent that delivered through "
                f"{CLAUDE_HANDBACK_TOOL} also had native stops held to the "
                "contract.",
            )
        return fires[-1]

    def _diagnostics_stop_fires(self, workspace: Path) -> list[dict]:
        """The recorded SubagentStop fires of 'tpu-diagnostics', in order."""
        return [
            e for e in self._recorded_events(workspace, "SubagentStop")
            if e.get("subagent") == "tpu-diagnostics"
        ]

    def _satisfies_diagnostics_result(self, fire: dict,
                                      test_pkg: Path) -> bool:
        """Whether a logged SubagentStop fire's output holds a valid result."""
        diagnostics_result_schema = json.loads(
            (test_pkg / DIAGNOSTICS_RESULT_SCHEMA).read_text(encoding="utf-8"))
        return any(
            not validate_instance(obj, diagnostics_result_schema)
            for obj in extract_all_json_objects(fire.get("output") or ""))

    def _assert_delivered_in_stdout(self, stdout: str, test_pkg: Path):
        """Assert the parent received a result satisfying the result contract.

        The stop log shows what the hooks checked; only the parent's output
        shows what the harness delivered. A refusal the harness ignored, say
        one shaped for the wrong event, passes the log check but not this.
        """
        self.assertTrue(
            self._satisfies_diagnostics_result({"output": stdout}, test_pkg),
            f"No result satisfying {DIAGNOSTICS_RESULT_SCHEMA} in the parent's "
            f"output:\n{stdout}")

    @_needs_cli("claude-code")
    def test_claude_code_cli_subagent_handback_deny_e2e(self):
        """E2E for a refused hand-back: Claude Code must honour the hooks' tool denial, and the subagent retry until its result passes.

        Also the auto-mode delivery road as a whole: the subagent may deliver
        only through SubagentHandback, and the accepted call reaches the parent.
        """
        if not _claude_has_handback():
            self.skipTest(
                f"Claude Code < {CLAUDE_HANDBACK_MIN_VERSION} has no "
                f"{CLAUDE_HANDBACK_TOOL}")
        test_pkg = self._create_schema_verifier_test_package()
        cli_res, _run_id, workspace = self._setup_and_run_harness_cli(
            harness="claude-code",
            prompt=HANDBACK_DENY_TEST_PROMPT,
            name="claude_handback_deny",
            extra_args=("--permission-mode", "auto"),
            timeout=240,
            package_dir=test_pkg,
        )
        self._assert_checked_by_one_road(workspace, require_handback=True)
        # Every fire is a hand-back, the assertion above guarantees it.
        handbacks = self._diagnostics_stop_fires(workspace)
        self.assertGreaterEqual(
            len(handbacks), 2,
            f"Expected a refused hand-back and a retry: {handbacks}")
        self.assertFalse(
            self._satisfies_diagnostics_result(handbacks[0], test_pkg),
            f"The first hand-back was meant to fail: {handbacks[0]}")
        self.assertTrue(
            self._satisfies_diagnostics_result(handbacks[-1], test_pkg),
            f"The retried hand-back did not pass: {handbacks[-1]}")
        # Only the parent's output shows Claude Code honoured the denial.
        self._assert_delivered_in_stdout(cli_res.stdout, test_pkg)

    @_needs_cli("claude-code")
    def test_claude_code_cli_schema_verifier_e2e(self):
        self._run_harness_cli_schema_verifier_e2e("claude-code",
                                                  "claude_schema")

    @_needs_cli("antigravity")
    def test_antigravity_cli_schema_verifier_e2e(self):
        self._run_harness_cli_schema_verifier_e2e("antigravity", "ag_schema")

    @_needs_cli("cursor")
    def test_cursor_cli_schema_verifier_e2e(self):
        self._run_harness_cli_schema_verifier_e2e("cursor", "cursor_schema")

    @_needs_cli("codex")
    def test_codex_cli_schema_verifier_e2e(self):
        self._run_harness_cli_schema_verifier_e2e("codex", "codex_schema")

    def _for_each_policy(self, run: Callable[[str], None]) -> None:
        """Run a scenario under `deny` and under `allow`.

        Each policy gets a fresh central store: the setup asserts no run
        exists yet, and the failure-log check reads the first record.
        """
        for on_failure in ("deny", "allow"):
            with self.subTest(on_failure=on_failure):
                shutil.rmtree(self.central_home, ignore_errors=True)
                self.central_home.mkdir(parents=True)
                run(on_failure)

    def _create_hung_hook_test_package(self, on_failure: str) -> Path:
        """Copy the package and guard `shell` with a hook that never answers."""

        def add_hung_hook(_test_pkg: Path, pkg_manifest: dict) -> None:
            pkg_manifest.setdefault("hooks", []).append({
                "name":
                HUNG_HOOK_NAME,
                "event":
                "PreToolUse",
                "matchers": ["shell"],
                "command":
                f"cat > {HUNG_HOOK_PAYLOAD}; sleep 30",
                "timeout":
                HUNG_HOOK_TIMEOUT_S,
                "on_failure":
                on_failure,
                "visibility":
                "external",
            })

        return self._copy_package(f"pkg_hung_hook_{on_failure}", add_hung_hook)

    def _run_harness_cli_hung_hook_e2e(self,
                                       harness: str,
                                       prefix: str,
                                       extra_args: tuple[str, ...] = ()):
        """A hook that hangs: dispatch's verdict on it must reach the harness.

        hook-dispatch settles a hook that outruns its timeout by its
        `on_failure` policy, but finishes a little after that timeout. With
        the harness registered on the same deadline, it killed dispatch
        first, dropped the deny, and -- on Claude Code, at least -- ran the
        guarded command. The compiler now gives the harness headroom.

        With `deny`, the guarded command must not run and the timeout must be
        the reason given. With `allow`, it must run, and the failure must
        still be on record in the workspace's hook failure log.
        """
        self._for_each_policy(lambda on_failure: self._hung_hook_once(
            harness, f"{prefix}_hung_hook_{on_failure}", extra_args, on_failure
        ))

    def _hung_hook_once(self, harness: str, name: str,
                        extra_args: tuple[str, ...], on_failure: str):
        test_pkg = self._create_hung_hook_test_package(on_failure)
        cli_res, _run_id, workspace = self._setup_and_run_harness_cli(
            harness=harness,
            prompt=HUNG_HOOK_TEST_PROMPT,
            name=name,
            extra_args=extra_args,
            timeout=120,
            package_dir=test_pkg,
        )

        # The attempt was made and the hung hook fired for it: the event it
        # saved names the command. Without this, a model that never called
        # the tool would pass.
        payload_file = workspace / HUNG_HOOK_PAYLOAD
        self.assertTrue(
            payload_file.is_file(),
            f"{HUNG_HOOK_NAME} never fired; nothing was tested. Output "
            f"was:\n{cli_res.stdout}")
        self.assertIn(
            HUNG_HOOK_SENTINEL, payload_file.read_text(encoding="utf-8"),
            f"{HUNG_HOOK_NAME} fired, but not for the touch it was meant to "
            f"guard. Output was:\n{cli_res.stdout}")

        # Dispatch, not the harness, settled the hook, and said so, in the
        # central store the harness inherited through TPU_AGENT_SUITE_HOME.
        failures = [
            r for r in read_jsonl(self.central_home / "logs" /
                                  "hook_failures.jsonl")
            if r.get("hook") == HUNG_HOOK_NAME
        ]
        self.assertTrue(
            failures,
            f"No hook failure was recorded for {HUNG_HOOK_NAME}. Output "
            f"was:\n{cli_res.stdout}")
        self.assertEqual(failures[0]["kind"], "timeout")
        self.assertEqual(failures[0]["decision"], on_failure)

        if on_failure == "allow":
            # The tool ran: failing open let the agent carry on.
            self.assertTrue(
                (workspace / HUNG_HOOK_SENTINEL).exists(),
                f"{HUNG_HOOK_SENTINEL} was not created: {harness} did not "
                f"run the command the failing-open hook was guarding. Output "
                f"was:\n{cli_res.stdout}")
            return

        # The tool never ran: the deny was honoured.
        self.assertFalse(
            (workspace / HUNG_HOOK_SENTINEL).exists(),
            f"{HUNG_HOOK_SENTINEL} was created: {harness} ran the command "
            f"the hung hook was guarding. Output was:\n{cli_res.stdout}")

        # And it was dispatch's deny, not the harness giving up on the hook
        # or refusing the command for a reason of its own.
        self.assertIn(
            f"timed out after {HUNG_HOOK_TIMEOUT_S} seconds", cli_res.stdout,
            f"Expected {HUNG_HOOK_NAME}'s timeout reason in the output:\n"
            f"{cli_res.stdout}")

    @_needs_cli("claude-code")
    def test_claude_code_cli_hung_hook_e2e(self):
        self._run_harness_cli_hung_hook_e2e("claude-code", "claude",
                                            CLAUDE_ALLOW_TOUCH)

    @_needs_cli("antigravity")
    def test_antigravity_cli_hung_hook_e2e(self):
        self._run_harness_cli_hung_hook_e2e("antigravity", "ag")

    @_needs_cli("cursor")
    def test_cursor_cli_hung_hook_e2e(self):
        self._run_harness_cli_hung_hook_e2e("cursor", "cursor")

    @_needs_cli("codex")
    def test_codex_cli_hung_hook_e2e(self):
        self._run_harness_cli_hung_hook_e2e("codex", "codex")

    def _create_missing_runtime_test_package(self, on_failure: str) -> Path:
        """Copy the package; guard `shell` with a hook of the given policy.

        Also add a Stop hook with the default policy. Neither will ever run:
        the test removes the dispatcher before the session starts.
        """

        def add_hooks(_test_pkg: Path, pkg_manifest: dict) -> None:
            pkg_manifest.setdefault("hooks", []).extend([{
                "name":
                MISSING_RUNTIME_GUARD_NAME,
                "event":
                "PreToolUse",
                "matchers": ["shell"],
                "command":
                "true",
                "on_failure":
                on_failure,
                "visibility":
                "external",
            }, {
                "name":
                MISSING_RUNTIME_STOP_NAME,
                "event":
                "Stop",
                "command":
                "true",
                "visibility":
                "external",
            }])

        return self._copy_package(f"pkg_missing_runtime_{on_failure}",
                                  add_hooks)

    def _run_harness_cli_missing_runtime_e2e(self,
                                             harness: str,
                                             prefix: str,
                                             extra_args: tuple[str, ...] = ()):
        """runtime.py gone from the store: the shell fallback alone answers.

        This is the bug's repro on a real session. Every hook command runs
        `python3 .../runtime.py`, which exits 2 on a file it cannot open;
        the compiled fallback turns that into the policy's code on the tool
        event and into 0 on Stop. With `allow` the guarded command must run;
        with `deny` it must not. Either way the session must end: the Stop
        hook carries the default `deny`, and a block there is what looped.
        """
        self._for_each_policy(lambda on_failure: self._missing_runtime_once(
            harness, f"{prefix}_missing_runtime_{on_failure}", extra_args,
            on_failure))

    def _missing_runtime_once(self, harness: str, name: str,
                              extra_args: tuple[str, ...], on_failure: str):
        test_pkg = self._create_missing_runtime_test_package(on_failure)

        def remove_runtime(_workspace: Path) -> None:
            runtimes = list((self.central_home /
                             "packages").glob("*/framework/runtime.py"))
            self.assertTrue(runtimes, "no installed runtime.py to remove")
            for runtime in runtimes:
                runtime.rename(runtime.with_suffix(".py.moved"))

        started = time.monotonic()
        cli_res, _run_id, workspace = self._setup_and_run_harness_cli(
            harness=harness,
            prompt=MISSING_RUNTIME_TEST_PROMPT,
            name=name,
            extra_args=extra_args,
            timeout=MISSING_RUNTIME_SESSION_TIMEOUT_S,
            package_dir=test_pkg,
            before_run=remove_runtime,
            expect_run=False,
        )
        elapsed = time.monotonic() - started

        sentinel = workspace / MISSING_RUNTIME_SENTINEL
        if on_failure == "allow":
            self.assertTrue(
                sentinel.exists(),
                f"{MISSING_RUNTIME_SENTINEL} was not created: {harness} "
                f"blocked the command although the guard fails open. Output "
                f"was:\n{cli_res.stdout}\n{cli_res.stderr}")
        else:
            self.assertFalse(
                sentinel.exists(),
                f"{MISSING_RUNTIME_SENTINEL} was created: {harness} ran the "
                f"command although the guard fails closed. Output was:\n"
                f"{cli_res.stdout}\n{cli_res.stderr}")

        # The session ended on its own well inside the timeout: the Stop
        # hook's default deny did not keep it going.
        self.assertLess(
            elapsed, MISSING_RUNTIME_SESSION_TIMEOUT_S * 0.75,
            f"{harness} session took {elapsed:.0f}s; a Stop hook with a dead "
            f"dispatcher may still be blocking the turn from ending. Output "
            f"was:\n{cli_res.stdout}\n{cli_res.stderr}")

    @_needs_cli("claude-code")
    def test_claude_code_cli_missing_runtime_e2e(self):
        self._run_harness_cli_missing_runtime_e2e("claude-code", "claude",
                                                  CLAUDE_ALLOW_TOUCH)

    @_needs_cli("antigravity")
    def test_antigravity_cli_missing_runtime_e2e(self):
        self._run_harness_cli_missing_runtime_e2e("antigravity", "ag")


if __name__ == "__main__":
    unittest.main()
