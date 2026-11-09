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

# The round finished (TP 2 x DP 4 won at the user's traffic and was
# re-profiled) and the ask is a custom kernel: does tpu-agent hand off to
# tpu-kernel-development with serving_round, building on the kept candidate
# with its own manifest and launch recipe?
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


CAND = "/workspace/optimization-heuristics/changes/m-tp2xdp4/deployments.json"


def request(args_json):
    """The kernel-request a delegation carried, as a dict when it parses."""
    try:
        prompt = json.loads(args_json).get("prompt", "")
        return json.loads(prompt) if isinstance(prompt, str) else prompt
    except (ValueError, AttributeError):
        return {}


reqs = [request(a) for a in to("tpu-kernel-development")]
raw = " ".join(to("tpu-kernel-development"))
sr = next((r.get("serving_round") for r in reqs if r.get("serving_round")), {})
sc = next((r.get("serving_config") for r in reqs if r.get("serving_config")), {})
CAND_RECIPE = "/workspace/optimization-heuristics/changes/m-tp2xdp4/launch_vllm.sh"
checks = {
    "kernel_delegated": bool(reqs),
    "serving_round_present": bool(sr) or "serving_round" in raw,
    "status_kept_candidate": (sr or {}).get("status") == "kept_candidate",
    "serving_config_is_kept_candidate":
    (sc or {}).get("deployments_json") == CAND,
    "recipe_matches_manifest": (sc or {}).get("launch_recipe") in (None,
                                                                 CAND_RECIPE),
    "targets_the_kept_profile": "b16-q1" in raw or "kept-m-tp2xdp4" in raw,
}
score = 1.0 if (checks["kernel_delegated"] and checks["serving_round_present"]
                and checks["serving_config_is_kept_candidate"]
                and checks["recipe_matches_manifest"]) else 0.0
write(score, checks, serving_round=sr, serving_config=sc)
PYEOF
