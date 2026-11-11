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

# Verifies that serving optimization heuristics on a disaggregated 1P1D
# GLM-5-FP8 deployment (TP=8, PCP=2, DCP=16, MTP=3 -> 4 decode tokens/req):
# 1. Actually ran. Detector IDs count only from the analyzer's own output,
#    never from answer.json alone or from files the agent could have written:
#    a. a successful tool call that is one plain run of the installed suite's
#       analyze.py with --events (`python3 <suite home>/packages/<package>/
#       scripts/optimization-heuristics/analyze.py ...`; VAR=value, env and
#       timeout prefixes, a leading `cd DIR &&` or `cd DIR` line, line
#       continuations and redirections are allowed, no other shell control
#       operator), whose output has the analyzer's closing
#       `Wrote .../findings.json and report.md` line. IDs are read from its
#       `<label>: D..., D...` lines only.
#    b. a successful delegation to tpu-diagnostics whose own result names the
#       findings.json the analyzer wrote in the agent suite's run directory.
#    Anything else, including a missing trajectory.json, scores 0.0 (fail
#    closed). Trajectory entries that are not JSON objects are skipped, so a
#    malformed trajectory scores 0.0 instead of crashing the verifier.
# 2. Surfaces D52_moe_tp_decode_row_limit (--enable-expert-parallel on decode)
#    at concurrency 20 (80 decode tokens -> 128-token bucket > 64-row limit).
# 3. Surfaces D18_attn_indexer with asymmetric decode-only DCP=8 at
#    concurrency 12 (48 decode tokens -> 64-token bucket) while preserving
#    PREFILL_DCP_SIZE=16 on the prefill engine so KV capacity is not halved;
#    the answer text has to state both sizes, not just mention prefill.
# 4. Rejects invalid TP=1 mesh layouts on vllm-torchtpu.
# 5. Submits no compute job: no `compute submit` or `compute run` (with or
#    without flags such as `--package <tag>` before the verb) and no
#    `kubectl apply`.
set -uo pipefail

LOGS="${LOGS_DIR:-/logs/verifier}"
mkdir -p "$LOGS"

python3 - "$LOGS" "${ANSWER_JSON_PATH:-/logs/artifacts/answer.json}" \
  "${AGENT_LOGS_DIR:-/logs/agent}" \
  "${TPU_AGENT_SUITE_RUNS:-/root/.tpu-agent-suite/runs}" \
  "${WORKSPACE_DIR:-/workspace}" \
  "${TPU_AGENT_SUITE_HOME:-/root/.tpu-agent-suite}" <<'PYEOF'
import json
import pathlib
import posixpath
import re
import shlex
import sys

logs, answer_path, agent_logs, runs, workspace = (
    pathlib.Path(p) for p in sys.argv[1:6])
suite_home = sys.argv[6].rstrip("/")
if not answer_path.is_file():
    fallback = workspace / "answer.json"
    if fallback.is_file():
        answer_path = fallback

D52 = "D52_moe_tp_decode_row_limit"
D18 = "D18_attn_indexer"
# The installer keeps each package at <suite home>/packages/<name>@<version>/
# (agent-suite runs mount it under /root/.tpu-agent-suite/packages/), and the
# compiled agents spell the analyzer ~/.tpu-agent-suite/packages/.../
# scripts/optimization-heuristics/analyze.py.
SUITE_PACKAGES = tuple(
    dict.fromkeys(f"{home}/packages/"
                  for home in (suite_home, "~/.tpu-agent-suite",
                               "$HOME/.tpu-agent-suite",
                               "${HOME}/.tpu-agent-suite")))
ANALYZER_SCRIPT = "/scripts/optimization-heuristics/analyze.py"
_CONTINUATION = re.compile(r"\\\r?\n")
_LEAD_CD = re.compile(r"\s*cd\s+(\S+)\s*(?:&&|\n)\s*")
_REDIRECT = re.compile(r"\s\d?>>?\s*[^\s;&|<>]+|\s\d?>&\d+")
_CONTROL = re.compile(r"[;&|<>`\n]|\$\(")
_ASSIGNMENT = re.compile(r"[A-Za-z_]\w*=\S*")
_DURATION = re.compile(r"\d+(?:\.\d+)?[smhd]?")
_PYTHON = re.compile(r"(?:\S*/)?python[0-9.]*")
_PYTHON_FLAGS = re.compile(r"-[BEIOSbqsu]+")
# analyze.py prints `<label>: D..., D...` per profile, then the Wrote line.
_ANALYZER_IDS = re.compile(
    r"^[^\n]*?: (D\d+_[a-z0-9_]+(?:, D\d+_[a-z0-9_]+)*)[ \t\r]*$", re.M)
_ANALYZER_WROTE = re.compile(
    r"^[ \t]*Wrote [^\n]*findings\.json and report\.md[ \t\r]*$", re.M)
# A compute job, with or without flags such as `--package <tag>` before the
# verb (compiled agents spell `compute --package <tag> run ...`).
_COMPUTE_JOB = re.compile(
    r"\bcompute(?:\s+--?[\w-]+(?:(?:=|\s+)(?!-)\S+)?)*\s+(?:submit|run)\b"
    r"|\bkubectl\s+apply\b")


def as_dict(val: object) -> dict:
    """`val` if it is a JSON object, else {} (a malformed field is empty)."""
    return val if isinstance(val, dict) else {}


def as_list(val: object) -> list:
    """`val` if it is a JSON array, else [] (a malformed field is empty)."""
    return val if isinstance(val, list) else []


def result_text(result: dict) -> str:
    """A tool result's text (a string or a list of content parts)."""
    content = result.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(p.get("text") or "") if isinstance(p, dict) else str(p)
            for p in content)
    return ""


def analyzer_argv(cmd: str) -> list[str] | None:
    """The arguments of one plain run of the suite's analyze.py, else None.

    Anything else in the same call (`;`, `&&`, `||`, `|`, `&`, a second line,
    command substitution) could print text that looks like the analyzer's, so
    such a call is no evidence. A relative script path counts only below a
    leading `cd` into the installed package.
    """
    body = _CONTINUATION.sub(" ", cmd)
    cwd = ""
    lead = _LEAD_CD.match(body)
    if lead:
        cwd = lead.group(1).strip("'\"")
        body = body[lead.end():]
    body = _REDIRECT.sub(" ", body).strip()
    if _CONTROL.search(body):
        return None
    try:
        words = shlex.split(body)
    except ValueError:
        return None
    while words:
        if words[0] == "env" or _ASSIGNMENT.fullmatch(words[0]):
            words = words[1:]
        elif words[0] == "timeout":
            words = words[1:]
            while words and words[0].startswith("-"):
                takes_value = words[0] in ("-s", "--signal", "-k",
                                           "--kill-after")
                words = words[2:] if takes_value else words[1:]
            if words and _DURATION.fullmatch(words[0]):
                words = words[1:]
        else:
            break
    if not words or not _PYTHON.fullmatch(words[0]):
        return None
    words = words[1:]
    while words and _PYTHON_FLAGS.fullmatch(words[0]):
        words = words[1:]
    if not words:
        return None
    script = words[0]
    if cwd and not script.startswith(("/", "~", "$")):
        script = posixpath.join(cwd, script)
    script = posixpath.normpath(script)
    if (not script.startswith(SUITE_PACKAGES)
            or not script.endswith(ANALYZER_SCRIPT)):
        return None
    if not any(w == "--events" or w.startswith("--events=")
               for w in words[1:]):
        return None
    return words


def findings_ids(path: pathlib.Path) -> set[str]:
    """Detector IDs in an analyzer findings.json (empty if it is not one)."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    ids = set()
    profiles = payload.get("profiles") if isinstance(payload, dict) else None
    for prof in profiles or []:
        result = prof.get("result") if isinstance(prof, dict) else None
        for finding in (result or {}).get("findings") or []:
            did = finding.get("detector_id") if isinstance(finding,
                                                           dict) else None
            if isinstance(did, str):
                ids.add(did)
    return ids


# Harbor's ATIF trajectory.json records each tool call and its result
# (Claude Code Agent/Task/Bash, Antigravity invoke_subagent/run_command).
# Claude Code's includes its subagents' steps; Antigravity's holds the main
# session only, so there a delegation counts through its result text alone.
traj_path = agent_logs / "trajectory.json"
try:
    steps = json.loads(traj_path.read_text(encoding="utf-8")).get("steps")
except (OSError, ValueError, AttributeError):
    steps = None
# A step, result or call that is not a JSON object, or a call ID that is not
# a string, is skipped: malformed entries are no evidence and never crash.
steps = [s for s in as_list(steps) if isinstance(s, dict)]

tool_calls_ok = 0
delegations = []
diagnostics_results = []
analyzer_calls = []
analyzer_ids: set[str] = set()
launched_compute = False
for step in steps:
    results = {
        r["source_call_id"]: r
        for r in as_list(as_dict(step.get("observation")).get("results"))
        if isinstance(r, dict) and isinstance(r.get("source_call_id"), str)
    }
    for call in as_list(step.get("tool_calls")):
        if not isinstance(call, dict):
            continue
        call_id = call.get("tool_call_id")
        res = results.get(call_id, {}) if isinstance(call_id, str) else {}
        extra = as_dict(res.get("extra"))
        if (extra.get("tool_result_is_error") is True
                or extra.get("status") in ("ERROR", "FAILED")):
            continue
        tool_calls_ok += 1
        fn = str(call.get("function_name") or "")
        args = as_dict(call.get("arguments"))
        named = []
        if fn in ("Agent", "Task"):
            named = [str(args.get("subagent_type") or "")]
        elif fn == "invoke_subagent":
            named = [
                str(sub.get("TypeName") or "")
                for sub in as_list(args.get("Subagents")) or [args]
                if isinstance(sub, dict)
            ]
        elif fn in ("Bash", "run_command"):
            cmd = str(args.get("command") or args.get("CommandLine") or "")
            if _COMPUTE_JOB.search(cmd):
                launched_compute = True
            text = result_text(res)
            if (analyzer_argv(cmd) is not None
                    and _ANALYZER_WROTE.search(text)):
                analyzer_calls.append(cmd[:2000])
                for m in _ANALYZER_IDS.finditer(text):
                    analyzer_ids |= set(m.group(1).split(", "))
        delegations += named
        if any(n.endswith("tpu-diagnostics") for n in named):
            diagnostics_results.append(result_text(res))

# A findings.json in the run directory counts only when a tpu-diagnostics
# delegation's own result names it, by its path or its path from runs/.
suite_findings = []
if diagnostics_results and runs.is_dir():
    for path in sorted(
            runs.glob("*/artifacts/**/optimization-heuristics/findings.json")):
        rel = f"/{runs.name}/{path.relative_to(runs).as_posix()}"
        if any(str(path) in t or rel in t for t in diagnostics_results):
            suite_findings.append(path)
for path in suite_findings:
    analyzer_ids |= findings_ids(path)
analyzer_ran = bool(analyzer_calls or suite_findings)


def write(score: float, checks: dict[str, bool], **extra: object) -> None:
    (logs / "reward.txt").write_text(f"{score}\n", encoding="utf-8")
    (logs / "reward.json").write_text(
        json.dumps({"reward": score}), encoding="utf-8"
    )
    (logs / "reward-details.json").write_text(
        json.dumps(
            {"checks": {k: int(v) for k, v in checks.items()}, **extra},
            indent=1,
        ),
        encoding="utf-8",
    )
    for k, v in checks.items():
        print(f"  {k:<30} {'PASS' if v else 'FAIL'}")


result: dict[str, object] = {}
if answer_path.is_file():
    try:
        loaded = json.loads(answer_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            result = loaded
    except ValueError:
        result = {}
if not result and steps:
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
                    result = loaded
                    break
            except ValueError:
                pass

raw_ids = result.get("detector_ids")
answer_ids = (
    {str(x).strip() for x in raw_ids}
    if isinstance(raw_ids, list)
    else set()
)


def as_bool(val: object) -> bool | None:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        low = val.strip().lower()
        if low in ("true", "yes", "1"):
            return True
        if low in ("false", "no", "0"):
            return False
    return None


def as_int(val: object) -> int | None:
    if isinstance(val, bool) or val is None:
        return None
    try:
        return int(str(val).strip())
    except ValueError:
        return None


def states_dcp(text: str, engine: str, size: int) -> bool:
    """Whether `text` gives `engine` (decode / prefill) a DCP of `size`."""
    dcp = r"(?:dcp|context[ _-]parallel)"
    return bool(re.search(
        rf"{engine}[^.\n]{{0,40}}?{dcp}[^.\n]{{0,25}}?\b{size}\b"
        rf"|{dcp}[^.\n]{{0,25}}?\b{size}\b[^.\n]{{0,40}}?{engine}",
        text, re.I))


ep_flag = as_bool(result.get("decode_enable_expert_parallel"))
decode_dcp = as_int(result.get("decode_dcp_size"))
prefill_dcp = as_int(result.get("prefill_dcp_size"))
allow_tp1 = as_bool(result.get("allow_tp1_mesh"))
answer_txt = str(result.get("answer") or "").strip()
ref_txt = str(result.get("reference") or "").strip()

low_ans = answer_txt.lower()
mentions_ep = (
    "enable-expert-parallel" in low_ans
    or "expert parallel" in low_ans
    or "expert-parallel" in low_ans
)
mentions_asym_dcp = (states_dcp(answer_txt, "decode", 8)
                     and states_dcp(answer_txt, "prefill", 16))

checks = {
    "analyzer_ran": analyzer_ran,
    "has_d52": D52 in analyzer_ids and D52 in answer_ids,
    "has_d18": D18 in analyzer_ids and D18 in answer_ids,
    "enables_decode_ep": ep_flag is True,
    "asymmetric_decode_dcp_8": decode_dcp == 8,
    "preserves_prefill_dcp_16": prefill_dcp == 16,
    "rejects_tp1_mesh": allow_tp1 is False,
    "has_answer_and_reference": bool(
        answer_txt and ref_txt and mentions_ep and mentions_asym_dcp
    ),
    "no_compute_job_submitted": not launched_compute,
}

score = 1.0 if all(checks.values()) else 0.0
write(
    score,
    checks,
    analyzer_detector_ids=sorted(analyzer_ids),
    answer_detector_ids=sorted(answer_ids),
    analyzer_calls=analyzer_calls,
    suite_findings=[str(p) for p in suite_findings],
    decode_enable_expert_parallel=ep_flag,
    decode_dcp_size=decode_dcp,
    prefill_dcp_size=prefill_dcp,
    allow_tp1_mesh=allow_tp1,
    trajectory_present=traj_path.is_file(),
    tool_calls_count=tool_calls_ok,
    delegations=[d for d in delegations if d],
)
PYEOF
