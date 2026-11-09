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

# With 8k/1k at 128 users named as the traffic, the rule keeps TP2xDP4:
# throughput +27.6% there, a loss at 1k/1k allowed and named.
set -uo pipefail
LOGS="${LOGS_DIR:-/logs/verifier}"
mkdir -p "$LOGS"
python3 - "$LOGS" "${AGENT_LOGS_DIR:-/logs/agent}" <<'PYEOF'
import json, pathlib, re, sys

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
        })[:600]) for name, prompt in subs if isinstance(name, str) and name]


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


reports = []
for f in pathlib.Path("/root/.tpu-agent-suite/runs").glob(
        "*/artifacts/optimization-heuristics/report.json"):
    try:
        reports.append(json.loads(f.read_text("utf-8")))
    except ValueError:
        pass
result = {}
a = pathlib.Path("/logs/artifacts/answer.json")
if a.is_file():
    try:
        result = json.loads(a.read_text("utf-8"))
    except ValueError:
        result = {}
decision = re.sub(r"\s+", " ", str(result.get("decision") or "")).strip().lower()
answer = str(result.get("answer") or "")
if not decision:
    # No answer file: the agent may give the decision in its final message
    # instead; take it from there.
    # The agent's last message in the trajectory (both harnesses).
    texts = [
        s.get("message") for s in steps
        if s.get("source") == "agent" and isinstance(s.get("message"), str)
        and s.get("message").strip()
    ]
    final = texts[-1] if texts else ""
    m = re.search(r"decision\W{0,6}\s*keep\s+(m-tp2xdp4|(?:the\s+)?baseline)",
                  final, re.I)
    if m:
        decision = "keep " + m.group(1).lower().replace("the ", "")
        answer = answer or final
checks = {
    "report_written": bool(reports),
    "decision_keep_m_tp2xdp4": decision == "keep m-tp2xdp4",
    "answer_has_both_points": ("8192" in answer or "8k" in answer.lower())
    and ("1024:1024" in answer or "1k/1k" in answer.lower()),
}
score = 1.0 if (checks["decision_keep_m_tp2xdp4"]) else 0.0
write(score, checks, decision=decision)
PYEOF
