#!/usr/bin/env python3
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
"""Contract enforcement through the hooks, against a real installed package.

The unit tests in `test_runtime.py` call the checkers directly, and the e2e
tests in `test_hooks_e2e.py` drive a real harness CLI and are skipped unless
`RUN_HARNESS_CLI_E2E=1`. Between those sits the layer that actually decides
whether a contract holds in production: `validate-delegate-request` and
`validate-subagent-result`, dispatched by the harness on every delegation.

This file covers that layer with no harness and no accelerator -- compile,
install into a temporary store, then hand `builtin_validate_schema` the
canonical payloads the harness would give it.

It exists because a bug hid exactly here. `check_status_payload` was wired
into `validate_json_file` only, so a result claiming an outcome while carrying
none of its measurements passed the SubagentStop hook while the CLI validator
rejected it -- and `validate_json_file` is a call an agent chooses to make,
while the hook is the one that always fires. Every unit test passed throughout.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
FRAMEWORK = REPO_ROOT / "framework"
PACKAGE = REPO_ROOT / "package"

RUN_ID = "20260923T101500Z-abcdef123456"
ARTIFACT = "artifacts/optimized.py"


def _load_installed_runtime(runtime_path: pathlib.Path):
    """Import an installed runtime.py together with its own `framework` package.

    The source tree's `framework` is already in `sys.modules` by the time this
    runs (this test module lives in it), so the installed runtime's
    `from framework import ...` would resolve to the source copies. Set those
    modules aside while importing, then put them back so other tests are
    unaffected. The installed modules stay reachable through the runtime.
    """
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "framework" or name.startswith("framework.")
    }
    saved_path = list(sys.path)
    for name in saved_modules:
        del sys.modules[name]
    sys.path.insert(0, str(runtime_path.parents[1]))
    try:
        spec = importlib.util.spec_from_file_location("installed_runtime",
                                                      runtime_path)
        assert spec and spec.loader
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
    finally:
        for name in [
                n for n in sys.modules
                if n == "framework" or n.startswith("framework.")
        ]:
            del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path
    return runtime


class TestContractHooks(unittest.TestCase):
    """Delegation requests and subagent results, judged by the real hook."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(cls._tmp.name)
        cls.home = root / "home"
        workspace = root / "ws"
        workspace.mkdir(parents=True)
        dist = root / "dist"

        env = {**os.environ, "TPU_AGENT_SUITE_HOME": str(cls.home)}
        compile_res = subprocess.run(
            [
                sys.executable,
                str(FRAMEWORK / "compiler.py"), "translate",
                str(PACKAGE), "--harness", "claude-code", "--output",
                str(dist)
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
        )
        if compile_res.returncode != 0:
            raise unittest.SkipTest(
                f"compiler failed, cannot exercise hooks:\n{compile_res.stderr[-2000:]}"
            )

        install_res = subprocess.run(
            [
                sys.executable,
                str(FRAMEWORK / "installer.py"), "install",
                str(dist), "--destination",
                str(workspace), "--non-interactive", "--skip-harness-check",
                "--relocated-store"
            ],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            env=env,
        )
        if install_res.returncode != 0:
            raise unittest.SkipTest(
                f"installer failed, cannot exercise hooks:\n{install_res.stderr[-2000:]}"
            )

        runtimes = sorted(
            (cls.home / "packages").glob("*/framework/runtime.py"))
        if not runtimes:
            raise unittest.SkipTest("installer placed no runtime to dispatch")

        # Import the *installed* runtime, not the source tree's. The hook runs
        # from the installed copy in production, so a bug that survives the
        # build is only visible from there.
        cls.runtime = _load_installed_runtime(runtimes[0])
        installed_framework = runtimes[0].parent.resolve()
        for module in (cls.runtime, cls.runtime.agent_contracts):
            if installed_framework not in pathlib.Path(
                    module.__file__).resolve().parents:
                raise AssertionError(
                    f"{module.__name__} loaded from {module.__file__}, not the"
                    f" installed copy in {installed_framework}")

        cls.workspace = workspace
        cls._prev_home = os.environ.get("TPU_AGENT_SUITE_HOME")
        os.environ["TPU_AGENT_SUITE_HOME"] = str(cls.home)

    @classmethod
    def tearDownClass(cls) -> None:
        if getattr(cls, "_prev_home", None) is None:
            os.environ.pop("TPU_AGENT_SUITE_HOME", None)
        else:
            os.environ["TPU_AGENT_SUITE_HOME"] = cls._prev_home
        cls._tmp.cleanup()

    def _request(self, agent: str, payload: dict | str) -> str:
        # A dict is encoded as the JSON payload a precise caller would send; a
        # str is passed through as the prose the kernel-worker actually dispatches.
        # Encoding the prose would wrap it in a JSON string literal, which the
        # hook reads as "no JSON object here" -- the opposite of the test.
        prompt = payload if isinstance(payload, str) else json.dumps(payload)
        decision = self.runtime.agent_contracts.builtin_validate_schema(
            "tpu-agent-suite", self.workspace, {
                "event": "PreToolUse",
                "tool": "delegate",
                "subagent": agent,
                "args": {
                    "prompt": prompt
                },
                "context": {
                    "run_id": RUN_ID
                },
            })
        return decision.get("decision", "allow")

    def _result(self, agent: str, payload: dict) -> str:
        decision = self.runtime.agent_contracts.builtin_validate_schema(
            "tpu-agent-suite", self.workspace, {
                "event": "SubagentStop",
                "subagent": agent,
                "output": "```json\n" + json.dumps(payload) + "\n```",
            })
        return decision.get("decision", "allow")

    # -- requests -------------------------------------------------------------

    def test_requests_reject_bad_payloads(self):
        """Each row is a valid dispatch with exactly one defect."""
        kernel = self._dispatch("tpu-kernel-development", "speed up fa")
        no_kernel = dict(kernel)
        del no_kernel["base_kernel_path"]
        no_prompt = dict(kernel)
        del no_prompt["prompt"]
        cases = [
            # The loop improves on an existing kernel, so a file is not
            # optional. A bare prompt used to validate, and the loop then fell
            # back to `<work>/base.py` -- a file the caller never named and may
            # not have written. The failure surfaced stages later, as a planner
            # with nothing to read, or worse as a real speedup measured against
            # the wrong kernel.
            ("tpu-kernel-development", "names no kernel", no_kernel),
            # `minLength: 1` -- an empty string names no file either.
            ("tpu-kernel-development", "empty base_kernel_path", {
                **kernel, "base_kernel_path": ""
            }),
            # A budget the loop has no stop condition for must not
            # validate. The schema is closed precisely so a caller cannot
            # promise the user an exit the agent will never take.
            ("tpu-kernel-development", "field the loop cannot honour", {
                **kernel, "time_budget_minutes": 30
            }),
            ("tpu-kernel-development", "no prompt", no_prompt),
            # A value that is present and malformed resolves to the wrong
            # artifact root, so it fails here rather than silently.
            ("tpu-kernel-development", "malformed run_id", {
                **kernel, "run_id": "not-a-run-id"
            }),
        ]
        for agent, label, payload in cases:
            with self.subTest(agent=agent, case=label):
                self.assertEqual(self._request(agent, payload), "deny")

    def test_typed_requests_are_accepted(self):
        """Every optional field a caller's instructions send is declared."""
        cases = [
            ("tpu-kernel-development", {
                "prompt": "speed up fa",
                "run_id": RUN_ID,
                "base_kernel_path": "work/base.py",
                "target_op": "flash attention",
                "max_iterations": 8,
                "atol": 0.001,
                "rtol": 0.001,
            }),
            ("kernel-worker", {
                "prompt":
                "Run one iteration of the loop.",
                "run_id":
                RUN_ID,
                "iteration":
                1,
                "base_kernel_path":
                "work/base.py",
                "work_dir":
                f"/runs/{RUN_ID}/artifacts/tpu-kernel-development/k01",
            }),
            # The open `fields` takes whatever quantities a caller asks for.
            ("tpu-diagnostics", {
                "prompt": "Analyse the trace.",
                "run_id": RUN_ID,
                "traces": [{
                    "path": "/w/iter1/trace"
                }],
                "report_path": "/w/iter1/p.md",
                "fields": ["alu_utilization_percent", "duty_cycle_percent"],
                "top_ops": 30,
            }),
            # Two traces in one call -- the xprof-compare shape one path lost.
            ("tpu-diagnostics", {
                "prompt":
                "Which run has the shorter step?",
                "run_id":
                RUN_ID,
                "traces": [{
                    "path": "/p/a",
                    "label": "A"
                }, {
                    "path": "/p/b",
                    "label": "B"
                }],
                "workload":
                "/w/variant.py",
            }),
        ]
        for agent, payload in cases:
            with self.subTest(agent=agent, fields=sorted(payload)):
                self.assertEqual(self._request(agent, payload), "allow")

    # -- every handoff in the tpu-kernel-development workflow -----------------------------

    # What each caller says. Every dispatch goes out as a JSON object carrying
    # `prompt` and `run_id` -- prose alone is denied, because a subagent never
    # sees SessionStart and the delegation is its only source for the run it
    # must write under. `DISPATCH_EXTRAS` adds whatever else a callee requires.
    WORKFLOW = [
        ("tpu-agent", "tpu-kernel-development",
         "Make the paged-attention Pallas kernel faster on v6e."),
        ("tpu-kernel-development", "kernel-worker",
         "Run one iteration of the loop."),
        ("tpu-kernel-development", "tpu-diagnostics",
         "Analyse the XProf trace at /w/iter1/trace; write /w/iter1/p.md."),
        ("kernel-worker", "analyze-cuda-reference",
         "Analyze the reference kernel at /w/flash.cu for /w (state file:"
         " /w/state.json)."),
    ]

    # Fields a callee requires beyond `prompt` and `run_id`.
    DISPATCH_EXTRAS = {
        "tpu-kernel-development": {
            "base_kernel_path": "work/base.py"
        },
        "kernel-worker": {
            "work_dir": f"/runs/{RUN_ID}/artifacts/tpu-kernel-development/k01"
        },
    }

    def _dispatch(self, callee: str, prompt: str) -> dict:
        payload = {"prompt": prompt, "run_id": RUN_ID}
        payload.update(self.DISPATCH_EXTRAS.get(callee, {}))
        return payload

    def test_every_workflow_handoff_is_accepted_as_dispatched(self):
        """Each dispatch, shaped as the instructions tell the caller to send it.

        These were denied for a long time. The delegation hook treats `prompt`
        as routing metadata and strips it unless the target schema declares it
        as a property -- so a schema without `prompt` was handed an empty
        object and failed on every required field it had.

        Declaring `prompt` fixed that; dropping `run_id` from `required` went
        too far the other way. Every stage now requires both, so the callers
        send JSON, and this asserts each edge is still reachable first time.
        """
        for caller, callee, prompt in self.WORKFLOW:
            with self.subTest(edge=f"{caller} -> {callee}"):
                self.assertEqual(
                    self._request(callee, self._dispatch(callee, prompt)),
                    "allow")

    def test_a_dispatch_without_run_id_is_denied(self):
        """The regression this pass fixes.

        Six result schemas require `run_id`. A subagent never sees
        SessionStart, so the delegation is its only source for the value:
        dispatched without one it cannot write an artifact under the right run
        and cannot return a valid result. The denial has to land on the caller,
        which holds the run_id and can resend, rather than on the subagent that
        can no longer recover it.
        """
        for caller, callee, prompt in self.WORKFLOW:
            payload = self._dispatch(callee, prompt)
            payload.pop("run_id")
            accepts = self.contract_of(callee).get("accepts") or ""
            if not accepts.endswith(".json"):
                continue  # no request schema -- nothing to enforce
            with self.subTest(edge=f"{caller} -> {callee}"):
                self.assertEqual(self._request(callee, payload), "deny")
        # Bare prose is validated as `{prompt}` and denied the same way --
        # with no `accepts` the hook used to allow it.
        with self.subTest(edge="prose -> tpu-diagnostics"):
            self.assertEqual(
                self._request("tpu-diagnostics",
                              "Analyse the XProf trace at /w/iter1/trace."),
                "deny")

    def contract_of(self, agent: str) -> dict:
        """The built contract for an agent, as the harness resolves it."""
        build = json.loads(
            next((self.home /
                  "packages").glob("*/tpu-agent-suite-build.json")).read_text(
                      encoding="utf-8"))
        return build["contracts"][agent]

    # -- results ----------------------------------------------------------------

    def test_claimed_outcome_without_its_payload_is_denied_at_the_hook(self):
        """The regression this file exists for: `status_requires` at the hook.

        Each payload satisfies its schema's shape -- only the conditional
        payload rule can catch it. Stage schemas have no `artifacts` property
        at all, so the completed-artifact rule never applies to them and this
        rule is the only guard.
        """
        reported = {
            "status": "reported",
            "summary": "done",
            "run_id": RUN_ID,
            "profiling_report_artifact": "/w/iter1/p.md",
            "xprof_version": "xprof 2.24.0",
            "traces": [{
                "path": "/w/iter1/trace",
                "captured": False
            }],
        }
        without_report = dict(reported)
        del without_report["profiling_report_artifact"]
        cases = [
            ("tpu-kernel-development", "optimized, no measurements", {
                "status": "optimized",
                "run_id": RUN_ID,
                "artifacts": [ARTIFACT],
                "summary": "went well",
            }),
            ("analyze-cuda-reference", "summarized, no payload", {
                "status": "summarized",
                "run_id": RUN_ID
            }),
            ("kernel-worker", "optimized, no measurements", {
                "status": "optimized",
                "run_id": RUN_ID,
                "iteration": 1
            }),
            ("tpu-diagnostics", "no report", without_report),
            # A report that read nothing is not a report; [] counts as absent.
            ("tpu-diagnostics", "no traces", {
                **reported, "traces": []
            }),
            ("tpu-diagnostics", "failed without a reason", {
                "status": "failed",
                "summary": "x",
                "run_id": RUN_ID
            }),
        ]
        for agent, label, payload in cases:
            with self.subTest(agent=agent, case=label):
                self.assertEqual(self._result(agent, payload), "deny")

    def test_an_optimized_kernel_must_point_at_a_durable_artifact(self):
        """The rule that renaming the status could have switched off silently.

        `check_completed_has_artifact` -- 'a completed result must carry an
        artifact' -- fires only on the literal status `completed`. Every
        measurement below is present; only the artifacts are missing or empty.
        """
        measured = {
            "status": "optimized",
            "run_id": RUN_ID,
            "slot": "k01",
            "base_kernel_path": "/workspace/work/base.py",
            "best_kernel_artifact": ARTIFACT,
            "best_latency_ms": 1.83,
            "speedup_factor": 2.31,
            "iterations_run": 3,
        }
        for label, artifacts in (("omitted", None), ("empty", [])):
            with self.subTest(artifacts=label):
                payload = dict(measured)
                if artifacts is not None:
                    payload["artifacts"] = artifacts
                self.assertEqual(
                    self._result("tpu-kernel-development", payload), "deny")

        self.assertEqual(
            self._result("tpu-kernel-development", {
                **measured, "artifacts": [ARTIFACT]
            }), "allow")

    def test_honest_results_are_allowed(self):
        """An honest failure must stay representable for every agent, and
        XProf's own strings and shapes are not the framework's to constrain."""

        def report(status, **fields):
            return dict(status=status, run_id=RUN_ID, **fields)

        diag = "tpu-diagnostics"
        # XProf's "N/A" and null metric values pass through.
        metrics = dict(peak_hbm_gib=dict(value="N/A",
                                         source_tool="get_kpi_metrics"),
                       alu_utilization_percent=dict(
                           value=None,
                           source_tool="get_llo_analysis",
                           tool_status="UNAVAILABLE",
                           tool_reason="LLO_DATA_ABSENT"))
        iteration = dict(iteration=1, compile_ok=False, test_ok=False)
        cases = [
            ("tpu-kernel-development",
             report("failed",
                    failure_reason="no_device",
                    iterations=[iteration])),
            ("analyze-cuda-reference",
             report("failed", failure_reason="timeout")),
            ("kernel-worker",
             report("failed", iteration=3,
                    failure_reason="compilation_error")),
            (diag, report("invalid_input", summary="no traces, no workload")),
            (diag, report("needs_input", summary="workload builds no inputs")),
            (diag,
             report("failed",
                    summary="read did not finish",
                    failure_reason="read_timeout")),
            (diag,
             report("reported",
                    summary="read in 143s",
                    profiling_report_artifact="/w/iter1/p.md",
                    xprof_version="xprof 2.24.0",
                    traces=[dict(path="/w/t", captured=True,
                                 metrics=metrics)])),
        ]
        for agent, payload in cases:
            with self.subTest(agent=agent, status=payload["status"]):
                self.assertEqual(self._result(agent, payload), "allow")


if __name__ == "__main__":
    unittest.main()
