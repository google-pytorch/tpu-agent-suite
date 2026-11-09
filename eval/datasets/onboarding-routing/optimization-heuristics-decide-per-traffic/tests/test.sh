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

# No traffic named: hand off per regime, judged at each regime's peak point.
# On the measured Gemma-4 layouts (TP 8 baseline): 1k/1k takes TP 4 x DP 2
# (+3.0% at c128; TP 2 x DP 4 loses there), 8k/1k takes TP 2 x DP 4 (+57% at
# c128, the larger gain), and the single configuration is the one kept for
# 8k/1k, the lower-peak regime.
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


def norm(v):
    v = re.sub(r"\s+", " ", str(v or "")).strip().lower()
    return v.replace("the baseline", "baseline")


per = {k: norm(v) for k, v in (result.get("per_regime") or {}).items()}
single = norm(result.get("single_config"))
if not per:
    # No answer file: take the hand-off from the agent's final message.
    # The agent's last message in the trajectory (both harnesses).
    texts = [
        s.get("message") for s in steps
        if s.get("source") == "agent" and isinstance(s.get("message"), str)
        and s.get("message").strip()
    ]
    final = texts[-1] if texts else ""
    for regime in ("1024:1024", "8192:1024"):
        # A row of the per-regime table: "| 1024:1024 | keep baseline | ...".
        row = next((l for l in final.splitlines()
                    if re.match(r"\s*\|\s*" + regime + r"\b", l)), "")
        m = re.search(r"(m-tp\dxdp\d|baseline|tp ?8)", row, re.I)
        if m:
            got = m.group(1).lower()
            per[regime] = "keep baseline" if got in ("baseline", "tp8",
                                                      "tp 8") else "keep " + got
    m = re.search(
        r"(?:single|one configuration)[^\n]{0,160}?(m-tp\dxdp\d|baseline|tp ?8)",
        final, re.I)
    if m:
        got = m.group(1).lower()
        single = "keep baseline" if got in ("baseline", "tp8",
                                            "tp 8") else "keep " + got
checks = {
    "report_has_regimes": any(
        c.get("regimes") for r in reports for c in r.get("candidates", [])),
    "short_takes_tp4xdp2": per.get("1024:1024") == "keep m-tp4xdp2",
    "long_takes_tp2xdp4": per.get("8192:1024") == "keep m-tp2xdp4",
    "single_config_is_lower_peak_regime": single == "keep m-tp2xdp4",
}
score = 1.0 if (checks["short_takes_tp4xdp2"]
                and checks["long_takes_tp2xdp4"]) else 0.0
write(score, checks, per_regime=per, single_config=single)
PYEOF
