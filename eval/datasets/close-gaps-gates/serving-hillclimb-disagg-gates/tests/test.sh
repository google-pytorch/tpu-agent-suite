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

# Deterministic CPU verifier for disaggregated 1P1D serving hill-climb gates.
# Run it with the experimental package (PACKAGE_MANIFEST=package.experimental.json):
# its close-gaps skill installs hillclimb_cli.py, and check 7 requires the
# gates' own output in the trajectory, so the release package and the baseline
# arm score 0 by design. What it measures is whether the agent routes to the
# gates instead of answering from the inputs alone.
#   1. Baseline server metrics show decode saturated (waiting p50=4.0, running 15.8/16)
#      while prefill is idle (waiting p50=0, running 0.28) -> binding_leg == decode.
#   2. arm_prefill_drafter tunes only prefill (DRAFTER_TP=4) -> reject; the reason
#      must say the deployment is decode-limited (an off-leg arm).
#   3. arm_sym_dcp8 sets symmetric DCP_SIZE=8 -> reject; the reason must say it
#      halves the prefill KV / prefix cache.
#   4. arm_cap20_mtp3 runs 20 * (1 + 3) = 80 decode tokens/step, above the largest
#      ROW_BUCKETS entry (64) -> needs_kernel_extension (or reject); the reason must
#      cite 80 and the 64 / ROW_BUCKETS bucket limit.
#   5. arm_asym_dcp (PREFILL_DCP_SIZE=16, DECODE_DCP_SIZE=8, TPU_KV_STRIPED_LAYOUT=0,
#      INDEXER_KV_CACHE_MODE=has, DECODE_INTERLEAVE=8) -> launch & selected_arm ==
#      "arm_asym_dcp".
#   6. inflight_arm_aiperf.txt runs at 0.752x the baseline's tok/s/chip at 10:30
#      elapsed -> stop; the reason must cite that ratio (~0.75x / -24.8%).
#   7. The trajectory has a shell call running `hillclimb_cli.py preflight` or
#      `legs`, and one running `hillclimb_cli.py early-verdict`, whose tool
#      results hold the CLI's own output for these fixtures: its text header or
#      --json keys, plus a value only the fixtures produce (a preflight check
#      name: off_leg, dcp_size_prefill_kv_halved or decode_tp_kernel; legs:
#      decode_limited; early-verdict: ratio 0.752 at 630 s). A hand-written
#      answer, an echoed command or echoed markers do not count.
#   8. No compute job was submitted: no `compute submit` or `compute run` (with
#      or without flags such as `--package <tag>` before the verb) and no
#      `kubectl apply`.
# A missing, unreadable or malformed trajectory scores 0; it does not crash the
# verifier.
set -uo pipefail
LOGS="${LOGS_DIR:-/logs/verifier}"
ANSWER="${ANSWER_JSON:-/logs/artifacts/answer.json}"
mkdir -p "$LOGS"

python3 - "$LOGS" "$ANSWER" "${AGENT_LOGS_DIR:-/logs/agent}" <<'PYEOF'
import json
import pathlib
import re
import sys

logs = pathlib.Path(sys.argv[1])
answer_path = pathlib.Path(sys.argv[2])
agent_logs = pathlib.Path(sys.argv[3])

traj_path = agent_logs / "trajectory.json"


def load_steps(path: pathlib.Path) -> list:
    """The trajectory's steps; [] when it is missing, unreadable or malformed."""
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    raw = data.get("steps") if isinstance(data, dict) else None
    return [s for s in raw if isinstance(s, dict)] if isinstance(raw, list) else []


steps = load_steps(traj_path)

SHELL_TOOLS = {"bash", "run_command", "run_shell_command", "execute_command", "shell"}
# For each gate: the text only hillclimb_cli.py prints (its text header, or
# the keys of its --json payload), and the values only this task's fixtures
# produce. Every fixture pattern must match.
GATE_OUTPUT = {
    "preflight": (
        (("PREFLIGHT ",), ('"has_errors"', '"findings"')),
        (r"(?<!\w)(?:off_leg|dcp_size_prefill_kv_halved|decode_tp_kernel)(?!\w)",),
    ),
    "legs": (
        (("leg_verdict:",), ('"off_leg_warning"', '"prefills"')),
        (r"(?<!\w)decode_limited(?!\w)",),
    ),
    "early-verdict": (
        (("early_verdict:",), ('"target_ratio"', '"status"')),
        (r'(?:ratio is |"ratio": )0\.752\d*', r'(?:at 630s|"elapsed_s": 630)(?!\d)'),
    ),
}
# A compute job, with or without flags such as `--package <tag>` before the verb.
COMPUTE_JOB = re.compile(
    r"\bcompute(?:\s+--?[A-Za-z0-9_][\w-]*(?:(?:=|\s+)(?!-)\S+)?)*\s+(?:submit|run)\b"
    r"|\bkubectl\s+apply\b"
)


def result_text(result) -> str:
    content = result.get("content") if isinstance(result, dict) else None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(c.get("text") or "") if isinstance(c, dict) else str(c)
            for c in content
        )
    return ""


def call_command(call: dict) -> str:
    args = call.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            return args
    if not isinstance(args, dict):
        return ""
    return str(args.get("command") or args.get("CommandLine") or "")


def runs_gate(cmd: str, gate: str) -> bool:
    """`cmd` runs hillclimb_cli.py with `gate` after it, not just names it."""
    return bool(re.search(
        rf"hillclimb_cli\.py\b.*?(?<![\w-]){re.escape(gate)}(?![\w-])", cmd, re.S))


gates_run = set()
launched_compute = False
for step in steps:
    observation = step.get("observation")
    raw_results = observation.get("results") if isinstance(observation, dict) else None
    results = {
        r.get("source_call_id"): r
        for r in (raw_results if isinstance(raw_results, list) else [])
        if isinstance(r, dict) and isinstance(r.get("source_call_id"), str)
    }
    raw_calls = step.get("tool_calls")
    for call in raw_calls if isinstance(raw_calls, list) else []:
        if not isinstance(call, dict):
            continue
        fn = str(call.get("function_name") or "").lower()
        if fn not in SHELL_TOOLS:
            continue
        cmd = call_command(call)
        if COMPUTE_JOB.search(cmd):
            launched_compute = True
        call_id = call.get("tool_call_id")
        out = result_text(results.get(call_id) if isinstance(call_id, str) else None)
        for gate, (markers, fixture) in GATE_OUTPUT.items():
            if (runs_gate(cmd, gate)
                    and any(all(m in out for m in group) for group in markers)
                    and all(re.search(p, out) for p in fixture)):
                gates_run.add(gate)


def write(score: float, checks: dict, **extra) -> None:
    (logs / "reward.txt").write_text(f"{score}\n", encoding="utf-8")
    (logs / "reward.json").write_text(
        json.dumps({"reward": score}), encoding="utf-8"
    )
    (logs / "reward-details.json").write_text(
        json.dumps(
            {"checks": {k: int(bool(v)) for k, v in checks.items()}, **extra},
            indent=2,
        ),
        encoding="utf-8",
    )
    for k, v in checks.items():
        print(f"  {k:<34} {'PASS' if v else 'FAIL'}")


payload = {}
if answer_path.is_file():
    try:
        loaded = json.loads(answer_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            payload = loaded
    except ValueError:
        payload = {}
if not payload and steps:
    texts = [
        s.get("message")
        for s in steps
        if s.get("source") == "agent"
        and isinstance(s.get("message"), str)
        and s.get("message").strip()
    ]
    for msg in reversed(texts):
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", msg, re.S)
        if m:
            try:
                loaded = json.loads(m.group(1))
                if isinstance(loaded, dict):
                    payload = loaded
                    break
            except ValueError:
                pass

binding_leg = str(payload.get("binding_leg") or "").strip().lower()
raw_decisions = payload.get("arm_decisions")
arm_decisions = raw_decisions if isinstance(raw_decisions, dict) else {}
selected_arm = str(payload.get("selected_arm") or "").strip()
inflight_action = str(payload.get("inflight_action") or "").strip().lower()
inflight_reason = str(payload.get("inflight_reason") or "").strip().lower()


def _decision_and_reason(arm_id: str) -> tuple[str, str]:
    entry = arm_decisions.get(arm_id)
    if isinstance(entry, dict):
        return (
            str(entry.get("decision") or "").strip().lower(),
            str(entry.get("reason") or "").strip().lower(),
        )
    if isinstance(entry, str):
        return entry.strip().lower(), ""
    return "", ""


d_prefill, r_prefill = _decision_and_reason("arm_prefill_drafter")
d_sym, r_sym = _decision_and_reason("arm_sym_dcp8")
d_cap20, r_cap20 = _decision_and_reason("arm_cap20_mtp3")
d_asym, _ = _decision_and_reason("arm_asym_dcp")

# The mechanism each reason has to name, not just a word from the arm itself.
_BINDING = r"(?:saturat\w*|bottleneck\w*|bind\w*|bound|limit\w*)"
DECODE_LIMITED = re.compile(
    r"off[_ -]?leg|decode[_ -](?:limited|bound)"
    rf"|\bdecode\b(?:\s+\w+){{0,3}}\s+{_BINDING}"
    rf"|\b{_BINDING}(?:\s+\w+){{0,3}}\s+decode\b"
)
HALVES = re.compile(r"halv|\bhalf\b|\b50\s*%|\b0\.5\s*x|11\.89|5\.95")
KV_CACHE = re.compile(r"\bkv\b|cache|\bhbm\b")
ABOVE_64 = re.compile(r"\b64\b|row_buckets|bucket")
RATIO_075 = re.compile(r"\b0\.7[45]\d*|\b7[45](?:\.\d+)?\s*%|\b2[45](?:\.\d+)?\s*%")

checks = {
    "binding_leg_is_decode": binding_leg in ("decode", "decode_limited", "decode-limited"),
    "rejects_off_leg_prefill_arm": (
        d_prefill in ("reject", "rejected", "discard")
        and bool(DECODE_LIMITED.search(r_prefill))
    ),
    "rejects_symmetric_dcp8": (
        d_sym in ("reject", "rejected", "discard")
        and bool(HALVES.search(r_sym))
        and bool(KV_CACHE.search(r_sym))
    ),
    "flags_80_row_bucket_cap": (
        d_cap20
        in (
            "needs_kernel_extension",
            "needs-kernel-extension",
            "reject",
            "rejected",
            "microbench_first",
        )
        and bool(re.search(r"\b80\b", r_cap20))
        and bool(ABOVE_64.search(r_cap20))
    ),
    "selects_asymmetric_dcp_arm": (
        selected_arm == "arm_asym_dcp"
        and d_asym in ("launch", "keep", "accept", "run")
    ),
    "stops_losing_inflight_arm": (
        inflight_action in ("stop", "stop_loss", "cancel", "kill")
        and bool(RATIO_075.search(inflight_reason))
    ),
    "ran_preflight_or_legs_gate": bool(gates_run & {"preflight", "legs"}),
    "ran_early_verdict_gate": "early-verdict" in gates_run,
    "no_compute_job_submitted": not launched_compute,
}

score = 1.0 if all(checks.values()) else 0.0
write(
    score,
    checks,
    binding_leg=binding_leg,
    selected_arm=selected_arm,
    inflight_action=inflight_action,
    trajectory_present=traj_path.is_file(),
    gates_run=sorted(gates_run),
)
PYEOF
