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

# Single- and multi-host accuracy-check recipe for `compute submit`.
#
# On a single-host slice (TPU_NUM_HOSTS unset or 1), starts one vLLM server
# from vllm_server_cmd.txt, waits for /health, prints the harness and
# serving-stack versions, sends one smoke request, runs the lm_eval command in
# eval_cmd.txt verbatim through `eval`, prints the eval-summary lines, and
# exits with lm_eval's exit code.
#
# On a multi-host Indexed Job (TPU_NUM_HOSTS > 1, TPU_WORKER_ID and
# TPU_HEAD_HOST set by compute), every host runs vllm_server_cmd.txt (which
# uses serve_vllm.sh or its own multi-host launcher so host 0 serves HTTP and
# hosts 1..N-1 join as workers). Only host 0 checks /health, runs the smoke
# request and lm_eval, and packs outputs; when host 0 finishes (pass or fail),
# it opens a brief TCP done-beacon on EVAL_COORD_PORT before stopping its
# server so worker pods stop cleanly and exit 0 instead of hanging or failing
# the Indexed Job when the leader shuts down. A worker whose server dies before
# the leader signals done exits non-zero so the Job fails fast.
#
# Inputs (staged next to this recipe with --input):
#   vllm_server_cmd.txt  the server command, run with `bash -c`
#   eval_cmd.txt         the lm_eval command, run with `eval`
#
# Arguments (KEY=VALUE, after `--`). An unknown key is an error.
#   SERVED_MODEL_NAME      required; the --served-model-name of the server
#   EVAL_TASKS             default mmlu_pro
#   EVAL_LIMIT             default 100; "full" runs the whole task (no --limit)
#   SERVER_PORT            default 8000; must match --port in the server command
#   SERVER_WAIT_TIMEOUT_S  default 3600; seconds to wait for /health
#   LM_EVAL_VERSION        default 0.5.0.dev1; installed only if lm_eval is absent
#   EVAL_COORD_PORT        default 8791; leader done-beacon port on multi-host
#
# Optional environment variables (for tuning and unit tests):
#   EVAL_COORD_GRACE_S       default 60; seconds leader keeps done-beacon open
#   EVAL_COORD_POLL_S        default 2; worker beacon poll interval (seconds)
#   EVAL_COORD_EXIT_RETRIES  default 10; 1 s worker retries after server exit
#   SERVER_HEALTH_POLL_S     default 10; leader /health poll interval (seconds)
#
# Outputs (in the recipe's working directory on host 0, fetch with --output):
#   eval_out.tar.gz      the lm_eval --output_path tree (results_*.json, samples_*.jsonl)
#   server.log           the full vLLM server log

set -uo pipefail

WORKDIR="$(pwd)"
EVAL_OUTPUT_DIR="${WORKDIR}/eval_out"
SERVER_LOG="${WORKDIR}/server.log"
mkdir -p "${EVAL_OUTPUT_DIR}"
: > "${SERVER_LOG}"

SERVED_MODEL_NAME=""
EVAL_TASKS="mmlu_pro"
EVAL_LIMIT="100"
SERVER_PORT="8000"
SERVER_WAIT_TIMEOUT_S="3600"
LM_EVAL_VERSION="0.5.0.dev1"
EVAL_COORD_PORT="${EVAL_COORD_PORT:-8791}"

NUM_HOSTS="${TPU_NUM_HOSTS:-1}"
WORKER_ID="${TPU_WORKER_ID:-${JOB_COMPLETION_INDEX:-0}}"
HEAD_HOST="${TPU_HEAD_HOST:-}"
EVAL_COORD_GRACE_S="${EVAL_COORD_GRACE_S:-60}"
EVAL_COORD_POLL_S="${EVAL_COORD_POLL_S:-2}"
EVAL_COORD_EXIT_RETRIES="${EVAL_COORD_EXIT_RETRIES:-10}"
SERVER_HEALTH_POLL_S="${SERVER_HEALTH_POLL_S:-10}"

SERVER_PID=""
COORD_PID=""
COORD_READY_FILE="${WORKDIR}/.eval_coord_ready"
LEADER_ACTIVE=0
FINISHING=0
SIGNALLED=0

is_worker() {
  (( NUM_HOSTS > 1 )) && [[ "${WORKER_ID}" != "0" ]]
}

pack_outputs() {
  tar -czf "${WORKDIR}/eval_out.tar.gz" -C "${WORKDIR}" eval_out 2>/dev/null \
    || tar -czf "${WORKDIR}/eval_out.tar.gz" --files-from /dev/null
}

stop_server() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill -TERM -- "-${SERVER_PID}" 2>/dev/null || kill -TERM "${SERVER_PID}" 2>/dev/null
    for _ in $(seq 1 30); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-${SERVER_PID}" 2>/dev/null || true
  fi
}

stop_coord_beacon() {
  if [[ -n "${COORD_PID}" ]] && kill -0 "${COORD_PID}" 2>/dev/null; then
    kill -TERM "${COORD_PID}" 2>/dev/null || true
    for _ in $(seq 1 10); do
      kill -0 "${COORD_PID}" 2>/dev/null || break
      sleep 0.1
    done
    kill -KILL "${COORD_PID}" 2>/dev/null || true
  fi
  if [[ -n "${COORD_PID}" ]]; then
    wait "${COORD_PID}" 2>/dev/null || true
    COORD_PID=""
  fi
  rm -f "${COORD_READY_FILE}"
}

start_coord_beacon() {
  local expected=$((NUM_HOSTS - 1))
  rm -f "${COORD_READY_FILE}"
  # Detach stdout/stderr from the caller's log FIFO so the background beacon
  # can never keep a staging tee/FIFO open across pod teardown.
  python3 - "${EVAL_COORD_PORT}" "${expected}" "${EVAL_COORD_GRACE_S}" "${COORD_READY_FILE}" >/dev/null 2>&1 <<'PY' &
import select
import socket
import sys
import time

port = int(sys.argv[1])
expected = int(sys.argv[2])
grace = float(sys.argv[3])
ready_file = sys.argv[4]

srv = None
try:
    srv = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
    srv.bind(("::", port))
except OSError:
    if srv is not None:
        srv.close()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))

srv.listen(max(expected, 4))
srv.setblocking(False)
with open(ready_file, "w", encoding="utf-8") as fh:
    fh.write("ready\n")

acked = set()
deadline = time.monotonic() + grace
while len(acked) < expected and time.monotonic() < deadline:
    remaining = max(0.05, deadline - time.monotonic())
    ready, _, _ = select.select([srv], [], [], min(0.5, remaining))
    if not ready:
        continue
    try:
        conn, addr = srv.accept()
    except OSError:
        continue
    with conn:
        conn.settimeout(2.0)
        try:
            msg = conn.recv(64).decode("utf-8", "replace").strip()
        except OSError:
            msg = ""
        try:
            conn.sendall(b"DONE\n")
        except OSError:
            pass
        parts = msg.split()
        key = parts[1] if len(parts) >= 2 and parts[0] == "ACK" else f"anon-{addr[0]}-{len(acked)}"
        acked.add(key)

srv.close()
PY
  COORD_PID=$!
  # Wait briefly for the listening socket before stop_server tears down vLLM.
  for _ in $(seq 1 20); do
    [[ -f "${COORD_READY_FILE}" ]] && break
    kill -0 "${COORD_PID}" 2>/dev/null || break
    sleep 0.1
  done
}

worker_coord_done() {
  python3 - "${HEAD_HOST}" "${EVAL_COORD_PORT}" "${WORKER_ID}" <<'PY'
import socket
import sys

host = sys.argv[1].strip().strip("[]")
port = int(sys.argv[2])
worker_id = sys.argv[3]
try:
    with socket.create_connection((host, port), timeout=2.0) as sock:
        sock.settimeout(2.0)
        sock.sendall(f"ACK {worker_id}\n".encode("utf-8"))
        reply = sock.recv(32)
    sys.exit(0 if reply.startswith(b"DONE") else 1)
except OSError:
    sys.exit(1)
PY
}

finish() {
  local rc="$1"
  local signalled="${2:-0}"
  if (( signalled )); then
    SIGNALLED=1
  fi
  if (( FINISHING )); then
    if (( signalled )); then
      stop_coord_beacon
    fi
    return
  fi
  FINISHING=1
  if is_worker; then
    stop_server
    echo "eval worker exited rc=${rc}"
    exit "${rc}"
  fi
  pack_outputs
  if (( NUM_HOSTS > 1 && LEADER_ACTIVE && ! SIGNALLED )); then
    start_coord_beacon
  fi
  stop_server
  if [[ -n "${COORD_PID}" ]]; then
    wait "${COORD_PID}" 2>/dev/null || true
    stop_coord_beacon
  fi
  echo "eval exited rc=${rc}"
  exit "${rc}"
}

fail() {
  echo "ERROR: $*" >&2
  finish 1
}

trap 'finish 143 1' TERM
trap 'finish 130 1' INT

for kv in "$@"; do
  key="${kv%%=*}"
  value="${kv#*=}"
  if [[ "${kv}" != *=* ]]; then
    fail "argument '${kv}' is not KEY=VALUE"
  fi
  case "${key}" in
    SERVED_MODEL_NAME|EVAL_TASKS|EVAL_LIMIT|SERVER_PORT|SERVER_WAIT_TIMEOUT_S|LM_EVAL_VERSION|EVAL_COORD_PORT)
      printf -v "${key}" '%s' "${value}" ;;
    *)
      fail "unknown key '${key}'; accepted: SERVED_MODEL_NAME EVAL_TASKS EVAL_LIMIT SERVER_PORT SERVER_WAIT_TIMEOUT_S LM_EVAL_VERSION EVAL_COORD_PORT" ;;
  esac
done

[[ -n "${SERVED_MODEL_NAME}" ]] || fail "SERVED_MODEL_NAME is required"
[[ -s "${WORKDIR}/vllm_server_cmd.txt" ]] || fail "vllm_server_cmd.txt is missing or empty; pass it with --input"
[[ -s "${WORKDIR}/eval_cmd.txt" ]] || fail "eval_cmd.txt is missing or empty; pass it with --input"
if (( NUM_HOSTS > 1 )) && [[ -z "${HEAD_HOST}" ]]; then
  fail "TPU_NUM_HOSTS=${NUM_HOSTS} requires TPU_HEAD_HOST"
fi

VLLM_SERVER_CMD="$(cat "${WORKDIR}/vllm_server_cmd.txt")"
EVAL_CMD="$(cat "${WORKDIR}/eval_cmd.txt")"

# The eval command takes its endpoint inside --model_args; reject it before the
# server spends a model load if it cannot reach the server at all.
if [[ "${EVAL_CMD}" != *base_url* ]]; then
  fail "eval_cmd.txt does not mention base_url"
fi

# vLLM's 600 s engine-ready / 60 s iteration / 300 s execute-model defaults
# kill large-model TPU compiles before /health answers (same defaults as
# serve_vllm.sh; any value the caller exported or puts in vllm_server_cmd.txt
# still wins).
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-1800}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"

if is_worker; then
  echo "eval_recipe: worker host ${WORKER_ID}/${NUM_HOSTS} (leader ${HEAD_HOST}, coord port ${EVAL_COORD_PORT})"
  setsid bash -c "${VLLM_SERVER_CMD}" >>"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  while kill -0 "${SERVER_PID}" 2>/dev/null; do
    if worker_coord_done; then
      echo "eval_recipe: worker ${WORKER_ID} received leader done; stopping server"
      finish 0
    fi
    sleep "${EVAL_COORD_POLL_S}"
  done
  wait "${SERVER_PID}"
  wrc=$?
  # The leader may have stopped its server right as it opened the done beacon,
  # causing a headless/Ray worker to exit non-zero on peer disconnect. Check
  # the beacon briefly before treating the exit as an early crash.
  for _ in $(seq 1 "${EVAL_COORD_EXIT_RETRIES}"); do
    if worker_coord_done; then
      echo "eval_recipe: worker ${WORKER_ID} server exited (rc=${wrc}) after leader done; exiting 0"
      finish 0
    fi
    sleep 1
  done
  echo "eval_recipe: worker ${WORKER_ID} server exited early (rc=${wrc}); last 200 lines of server.log:"
  tail -n 200 "${SERVER_LOG}"
  if (( wrc == 0 )); then
    wrc=1
  fi
  echo "eval worker exited rc=${wrc}"
  exit "${wrc}"
fi

LEADER_ACTIVE=1

if [[ "${EVAL_LIMIT}" == "full" || -z "${EVAL_LIMIT}" ]]; then
  EVAL_LIMIT_ARG=""
else
  EVAL_LIMIT_ARG="--limit ${EVAL_LIMIT}"
fi
EVAL_BASE_URL="http://localhost:${SERVER_PORT}/v1/chat/completions"
export SERVED_MODEL_NAME EVAL_TASKS EVAL_LIMIT EVAL_LIMIT_ARG EVAL_BASE_URL EVAL_OUTPUT_DIR SERVER_PORT

if (( NUM_HOSTS > 1 )); then
  echo "eval_recipe: leader host 0/${NUM_HOSTS} (head ${HEAD_HOST}, coord port ${EVAL_COORD_PORT})"
fi
echo "effective SERVED_MODEL_NAME: ${SERVED_MODEL_NAME}"
echo "effective EVAL_TASKS: ${EVAL_TASKS}"
echo "effective EVAL_LIMIT: ${EVAL_LIMIT}"
echo "effective SERVER_PORT: ${SERVER_PORT}"
echo "effective VLLM_SERVER_CMD: ${VLLM_SERVER_CMD}"
echo "effective EVAL_CMD: ${EVAL_CMD}"

# Harness version. Install the pinned fallback only when the image lacks it.
if ! python3 -c "import lm_eval" 2>/dev/null; then
  echo "lm_eval not present; installing lm-eval[api]==${LM_EVAL_VERSION}"
  if command -v uv >/dev/null 2>&1; then
    uv pip install --system --no-config "lm-eval[api]==${LM_EVAL_VERSION}" || fail "lm_eval install failed"
  else
    python3 -m pip install "lm-eval[api]==${LM_EVAL_VERSION}" || fail "lm_eval install failed"
  fi
fi
echo "lm_eval version: $(python3 -c 'import importlib.metadata as m; print(m.version("lm_eval"))' 2>/dev/null || echo unknown)"
echo "vllm version: $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null || echo unknown)"
echo "vllm-torchtpu commit: ${VLLM_TORCHTPU_COMMIT:-unknown}"

# Server, in its own process group so that stop_server reaches its workers.
setsid bash -c "${VLLM_SERVER_CMD}" >>"${SERVER_LOG}" 2>&1 &
SERVER_PID=$!
start=$(date +%s)
while true; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "vllm-server exited before /health answered; last 200 lines of server.log:"
    tail -n 200 "${SERVER_LOG}"
    finish 1
  fi
  if curl -sf "http://localhost:${SERVER_PORT}/health" >/dev/null 2>&1; then
    break
  fi
  if (( $(date +%s) - start > SERVER_WAIT_TIMEOUT_S )); then
    echo "vllm-server not up after ${SERVER_WAIT_TIMEOUT_S}s; last 200 lines of server.log:"
    tail -n 200 "${SERVER_LOG}"
    finish 1
  fi
  sleep "${SERVER_HEALTH_POLL_S}"
done
echo "vllm-server is up after $(( $(date +%s) - start ))s"
echo "Eval endpoint: ${EVAL_BASE_URL}"

# One real answer before the run commits hours of TPU time.
python3 - "${EVAL_BASE_URL}" "${SERVED_MODEL_NAME}" <<'PY'
import json
import sys
import urllib.request

url, model = sys.argv[1], sys.argv[2]
body = json.dumps({
    "model": model,
    "messages": [{"role": "user", "content": "What is 17 * 23? Answer with the number only."}],
    "max_tokens": 512,
    "temperature": 0,
}).encode()
req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=600) as resp:
        out = json.load(resp)
except Exception as exc:  # report and continue; the eval decides
    print(f"smoke: request failed: {exc}")
    sys.exit(0)
choice = (out.get("choices") or [{}])[0]
msg = choice.get("message") or {}
usage = out.get("usage") or {}
print(f"smoke: finish_reason={choice.get('finish_reason')} completion_tokens={usage.get('completion_tokens')}")
content = msg.get("content")
if content is None:
    print("smoke: content=<null>")
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    print(f"smoke: content was null; reasoning={reasoning[:200]!r}")
else:
    print(f"smoke: content={content.strip()[:200]}")
PY

# The eval command runs verbatim. Nothing is appended to it.
eval "${EVAL_CMD}"
rc=$?

# Summary lines, from the results and samples files lm_eval wrote.
python3 - "${EVAL_OUTPUT_DIR}" <<'PY'
import glob
import json
import os
import sys

root = sys.argv[1]
results = sorted(glob.glob(os.path.join(root, "**", "results_*.json"), recursive=True),
                 key=os.path.getmtime)
samples = glob.glob(os.path.join(root, "**", "samples_*.jsonl"), recursive=True)

if samples:
    nulls = 0
    for path in samples:
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                resps = rec.get("resps") or []
                flat = [r for group in resps for r in (group if isinstance(group, list) else [group])]
                if not flat or any(r is None or (isinstance(r, str) and not r.strip()) for r in flat):
                    nulls += 1
    print(f"eval-summary: null-content answers={nulls}")
else:
    print("eval-summary: null-content answers=unknown (no samples_*.jsonl; run with --log_samples)")

if not results:
    print("eval-summary: no results_*.json written")
    sys.exit(0)

with open(results[-1]) as f:
    data = json.load(f)
n_samples = data.get("n-samples") or {}
versions = data.get("versions") or {}
groups = data.get("groups") or {}
res = data.get("results") or {}


def line(task, metrics):
    parts = []
    for key, value in metrics.items():
        if key == "alias" or "_stderr" in key:
            continue
        stderr = metrics.get(key.replace(",", "_stderr,", 1))
        if isinstance(stderr, (int, float)):
            parts.append(f"{key}={value} +- {stderr}")
        else:
            parts.append(f"{key}={value}")
    ns = n_samples.get(task) or {}
    sample_len = metrics.get("sample_len") or ns.get("effective")
    if sample_len is None and task in groups:
        sample_len = sum((n_samples.get(t) or {}).get("effective", 0)
                         for t in res if t != task and t.startswith(task))
    print(f"eval-summary: {task} {' '.join(parts)} sample_len={sample_len} version={versions.get(task)}")


heads = [t for t in groups] or list(res)[:1]
for task in heads:
    line(task, groups.get(task) or res.get(task) or {})
for task, metrics in res.items():
    if task not in heads:
        line(task, metrics)
PY

finish "${rc}"
