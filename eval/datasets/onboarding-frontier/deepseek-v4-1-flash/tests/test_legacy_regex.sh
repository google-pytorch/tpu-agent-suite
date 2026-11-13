#!/usr/bin/env bash
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

# Scores the checkable half of the brief. "Coherent output" and "the log reads
# as the story of how you got there" are not machine-checkable and are left to
# a reader; everything below is either present or it is not.
#
# There is no forced-pass switch. The brief asks for a running server, and a
# verifier that reports 1.0 without one would certify the failure it exists to
# catch.

set -uo pipefail

LOGS="${LOGS_DIR:-/logs/verifier}"
mkdir -p "$LOGS"
ARTIFACTS_DIR="${ARTIFACTS:-/logs/artifacts}"
WORKSPACE_DIR="${WORKSPACE:-/workspace}"
AGENT_DIR="${AGENT_LOGS_DIR:-/logs/agent}"

python3 - "$LOGS" "$ARTIFACTS_DIR" "$WORKSPACE_DIR" "$AGENT_DIR" <<'PYEOF'
import json, os, pathlib, re, subprocess, sys

logs, artifacts, workspace, agent_dir = (pathlib.Path(p) for p in sys.argv[1:5])
roots = [p for p in (artifacts, workspace) if p.is_dir()]
agent_text = ""
if agent_dir.is_dir():
    for f in sorted(agent_dir.rglob("*"))[:400]:
        if f.is_file():
            agent_text += f.read_text("utf-8", errors="ignore")

def files():
    for r in roots:
        for f in r.rglob("*"):
            if f.is_file() and ".git" not in f.parts:
                yield f

def text_of(f, cap=400_000):
    try:
        return f.read_text("utf-8", errors="ignore")[:cap]
    except OSError:
        return ""

corpus = {f: text_of(f) for f in list(files())[:800]}
blob = "\n".join(corpus.values()) + "\n" + agent_text
checks, notes = {}, {}

# 1. Artifacts collected under one directory rather than scattered.
dirs = {f.parent for f in corpus}
checks["artifacts_dir"] = bool(corpus) and len(corpus) >= 3
notes["artifacts_dir"] = f"{len(corpus)} files under {len(dirs)} dirs"

# 2. One file pinning the versions the brief enumerates. The image digest and
#    the HF revision are the two that cannot be guessed after the fact.
digest = re.search(r"sha256:[0-9a-f]{64}", blob)
hf_rev = re.search(r"\b[0-9a-f]{40}\b", blob)
vllm_v = re.search(r"vllm[^\n]{0,40}?(\d+\.\d+\.\d+)", blob, re.I)
for key, hit in (("pin_image_digest", digest), ("pin_hf_revision", hf_rev),
                 ("pin_vllm", vllm_v)):
    checks[key] = bool(hit)
    notes[key] = hit.group(0) if hit else "absent"

# Topology must come from the pinned-versions file the brief asks for, not the
# agent's transcript (which carries the prompt and kubectl output) and not any
# NxM string in a benchmark log. A pin file is an artifact that records an image
# digest; it must name the accelerator and state the topology.
ACCEL = re.compile(r"\b(tpu7x|tpu-v6e-slice|v6e|v7x?)\b", re.I)
TOPO = re.compile(r"topolog\w*\W{0,20}(\d+x\d+(?:x\d+)?)\b", re.I)
pin_files = [f for f, t in corpus.items()
             if artifacts in f.parents and re.search(r"sha256:[0-9a-f]{64}", t)]
pin_text = "\n".join(corpus[f] for f in pin_files)
accel_hit, topo_hit = ACCEL.search(pin_text), TOPO.search(pin_text)
checks["pin_topology"] = bool(accel_hit and topo_hit)
notes["pin_topology"] = (
    f"{accel_hit.group(0)} {topo_hit.group(1)} in "
    f"{', '.join(f.name for f in pin_files)}"
    if accel_hit and topo_hit else "absent from pin file")

# 3/4. The two tables the brief specifies, by their headers.
bench_hdr = re.search(
    r"\|[^\n]*TTFT[^\n]*\|[^\n]*ITL[^\n]*\|[^\n]*Throughput[^\n]*\|", blob, re.I)
fail_hdr = re.search(
    r"\|[^\n]*Attempt[^\n]*\|[^\n]*Outcome[^\n]*\|[^\n]*Resolution[^\n]*\|", blob, re.I)
checks["benchmark_table"] = bool(bench_hdr)
checks["failure_table"] = bool(fail_hdr)

# A header with no data row underneath is a template, not a result.
def rows_under(match):
    if not match:
        return 0
    tail = blob[match.end():match.end() + 4000].splitlines()
    n = 0
    for line in tail[1:]:
        if not line.strip().startswith("|"):
            break
        if re.search(r"\d", line):
            n += 1
    return n

checks["benchmark_rows"] = rows_under(bench_hdr) > 0
notes["benchmark_rows"] = f"{rows_under(bench_hdr)} measured rows"

# 5. Raw output retained, not only the summary the tables were built from.
raw = [f for f in corpus if re.search(r"raw|\.log$|\.jsonl$|serve|bench", f.name, re.I)]
checks["raw_output"] = len(raw) >= 2
notes["raw_output"] = f"{len(raw)} raw/log files"

# 6. Evidence a server answered over HTTP, rather than a claim that it did.
checks["server_response"] = bool(
    re.search(r"/v1/(chat/)?completions", blob)
    and re.search(r'"(choices|content|text)"\s*:', blob))

# 7. A history, not one commit at the end. The repo is wherever the agent put
# it -- usually a subdirectory of the artifacts, not the artifacts root.
ncommits = 0
seen = set()
for r in roots:
    for gitdir in [r / ".git", *r.glob("*/.git"), *r.glob("*/*/.git")]:
        repo = gitdir.parent
        if not gitdir.exists() or repo in seen:
            continue
        seen.add(repo)
        out = subprocess.run(["git", "-C", str(repo), "rev-list", "--count", "HEAD"],
                             capture_output=True, text=True)
        if out.returncode == 0:
            ncommits = max(ncommits, int(out.stdout.strip() or 0))
checks["commit_history"] = ncommits >= 3
notes["commit_history"] = f"{ncommits} commits"

# 8. The two paths the brief rules out. The nightly ships tpu_inference at
#    /root, so this looks for use by the agent, not mere presence on disk.
banned = re.search(r"^(?!.*#).*\b(import|from)\s+(torch_xla|tpu_inference)\b",
                   blob, re.M)
checks["no_banned_backend"] = not banned
notes["no_banned_backend"] = banned.group(0).strip()[:60] if banned else "clean"

# 9. Credentials and weights stay out of the artifacts.
leak = re.search(r"ghp_[A-Za-z0-9]{20,}|github_pat_|BEGIN [A-Z ]*PRIVATE KEY", blob)
weights = [f for f in corpus if f.suffix in (".safetensors", ".bin", ".pt")]
checks["no_secrets_committed"] = not leak and not weights
notes["no_secrets_committed"] = ("token-like string found" if leak else
                                 f"{len(weights)} weight files" if weights else "clean")

# The two prohibitions are gates, not points: a run that produced nothing
# satisfies both, and scoring them would pay it for the absence of work.
# Breaking either is disqualifying -- the brief calls reading pre-existing work
# a FAIL outright, and committing a credential is not offset by a good table.
GATES = ("no_banned_backend", "no_secrets_committed")
earned = [k for k in checks if k not in GATES]
score = round(sum(bool(checks[k]) for k in earned) / len(earned), 4)
gate_broken = [k for k in GATES if not checks[k]]
if gate_broken:
    score = 0.0

(logs / "reward.txt").write_text(f"{score}\n")
# Harbor reads every key of reward.json as a reward and fails the trial on one
# that is not a number, so the report beside the score goes to its own file.
(logs / "reward.json").write_text(json.dumps({"reward": score}))
(logs / "reward-details.json").write_text(json.dumps(
    {"scoring": "measured",
     "gates_broken": gate_broken,
     "checks": {k: int(bool(v)) for k, v in checks.items()},
     "notes": notes}, indent=1))

width = max(len(k) for k in checks)
for k, v in checks.items():
    kind = "gate" if k in GATES else "    "
    print(f"  {kind} {k:<{width}}  {'PASS' if v else 'FAIL'}   {notes.get(k, '')}")
if gate_broken:
    print(f"\nDisqualified: {', '.join(gate_broken)}")
print(f"\nVerifier final score: {score}  "
      f"({sum(bool(checks[k]) for k in earned)}/{len(earned)} scored checks)")
PYEOF
