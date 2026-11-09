#!/bin/bash
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

# A running baseline with no profile yet, and an optimize ask: does tpu-agent's
# first tpu-diagnostics request (the profiling gate) carry serving_candidates,
# so the levers and the candidates come back together?
set -uo pipefail
LOGS="${LOGS_DIR:-/logs/verifier}"
mkdir -p "$LOGS"
python3 - "$LOGS" "${AGENT_LOGS_DIR:-/logs/agent}" <<'PYEOF'
import json, pathlib, sys

logs, agent = (pathlib.Path(p) for p in sys.argv[1:3])
# A delegation counts once it started and its result was not an error. Read
# from Harbor's ATIF trajectory, which records both harnesses (as
# eval/verifiers/common/invoked_subagent.py does): Claude Code's Agent / Task
# calls (prompt in `prompt`) and Antigravity's invoke_subagent (one per
# `Subagents` entry, prompt in `Prompt`). Each is kept as
# {"subagent_type", "prompt"}, in the order the run made them.
traj = agent / "trajectory.json"
steps = (json.loads(traj.read_text("utf-8")).get("steps") or []
         if traj.is_file() else [])
delegations = []
for step in steps:
    results = {
        r.get("source_call_id"): r
        for r in (step.get("observation") or {}).get("results") or []
    }
    for call in step.get("tool_calls") or []:
        extra = (results.get(call.get("tool_call_id")) or {}).get("extra") or {}
        if (extra.get("tool_result_is_error") is True
                or extra.get("status") in ("ERROR", "FAILED")):
            continue
        args = call.get("arguments") or {}
        if call.get("function_name") in ("Agent", "Task"):
            subs = [(args.get("subagent_type"), args.get("prompt"))]
        elif call.get("function_name") == "invoke_subagent":
            subs = [(s.get("TypeName"), s.get("Prompt"))
                    for s in args.get("Subagents") or []]
        else:
            continue
        delegations += [(name, json.dumps({
            "subagent_type": name,
            "prompt": prompt
        })[:20000]) for name, prompt in subs if isinstance(name, str) and name]


def to(name):
    return [a for t, a in delegations if t.endswith(name)]


def write(score, checks, **extra):
    (logs / "reward.txt").write_text(f"{score}\n")
    (logs / "reward.json").write_text(json.dumps({"reward": score}))
    (logs / "reward-details.json").write_text(json.dumps(
        {"checks": {k: int(v) for k, v in checks.items()}, **extra},
        indent=1))
    for k, v in checks.items():
        print(f"  {k:<28} {'PASS' if v else 'FAIL'}")


diag = to("tpu-diagnostics")
checks = {
    "diagnostics_called": bool(diag),
    "first_request_has_candidates": bool(diag) and "serving_candidates" in diag[0],
    "first_request_has_server": bool(diag) and "golden-gemma4-a1b2c3d-serve" in diag[0],
    "no_kernel_development": not to("tpu-kernel-development"),
}
score = 1.0 if (checks["first_request_has_candidates"]
                and checks["first_request_has_server"]
                and checks["no_kernel_development"]) else 0.0
write(score, checks, diagnostics_calls=diag[:2])
PYEOF
