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
#
# Start `python3 -m sglang.launch_server` with its profiler armed, and serve
# the profile directory so a trace can be fetched while the server runs.
#
# Environment: SGLANG_PROFILE_DIR (default artifacts/sglang_profile),
#              SGLANG_PROFILE_FILES_PORT (default 8790),
#              SGLANG_PROFILE_LLO (default 1; 0 leaves LIBTPU_INIT_ARGS as is),
#              PYTHON_TRACER_LEVEL (default 0; a value the caller sets is kept),
#              SGLANG_WARMUP_TIMEOUT (default 3600),
#              SGLANG_WATCHDOG_TIMEOUT_S (default 3600, passed as
#                --watchdog-timeout when this SGLang has the flag),
#              SGLANG_DIST_TIMEOUT_S (default 1800, passed as --dist-timeout
#                when this SGLang has the flag);
#              a timeout the caller sets, or passes as a flag, is kept
# Usage, in launch_sglang.sh:
#   exec bash serve_sglang.sh "$(cat weights_path.txt)" --port 8000 ...
#   exec bash serve_sglang.sh --profile-dir /tmp/sglang_profile -- --model-path ...

set -eo pipefail

PROFILE_DIR="${SGLANG_PROFILE_DIR:-${VLLM_PROFILE_DIR:-$PWD/artifacts/sglang_profile}}"
FILES_PORT="${SGLANG_PROFILE_FILES_PORT:-${VLLM_PROFILE_FILES_PORT:-8790}}"

while [ $# -gt 0 ]; do
  case "$1" in
    --profile-dir)
      PROFILE_DIR="$2"
      shift 2
      ;;
    --profile-dir=*)
      PROFILE_DIR="${1#*=}"
      shift
      ;;
    --files-port)
      FILES_PORT="$2"
      shift 2
      ;;
    --files-port=*)
      FILES_PORT="${1#*=}"
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

PY="${TPU_AGENT_SUITE_PYTHON:-python3}"
PY="$(command -v "${PY/#\~/$HOME}" || true)"

mkdir -p "$PROFILE_DIR"
export SGLANG_TORCH_PROFILER_DIR="$PROFILE_DIR"
export PYTHON_TRACER_LEVEL="${PYTHON_TRACER_LEVEL:-0}"

# --xla_xprof_register_llo_debug_info puts the LLO source map in the trace,
# which `get_llo_analysis` and any kernel-level read need.
LLO_FLAG="--xla_xprof_register_llo_debug_info"
PROFILE_LLO="${SGLANG_PROFILE_LLO:-${VLLM_PROFILE_LLO:-1}}"
if [ "$PROFILE_LLO" = 0 ]; then
  echo "serve_sglang: LLO debug info off (SGLANG_PROFILE_LLO=0)" >&2
elif [[ " ${LIBTPU_INIT_ARGS:-} " == *" $LLO_FLAG"[=\ ]* ]]; then
  echo "serve_sglang: keeping the caller's $LLO_FLAG" >&2
else
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+$LIBTPU_INIT_ARGS }$LLO_FLAG=true"
  echo "serve_sglang: LLO debug info on (LIBTPU_INIT_ARGS=$LIBTPU_INIT_ARGS)" >&2
fi

# SGLang's timeouts are sized for GPUs. Compiling a large model for TPU runs
# past the 600 s warmup and 300 s watchdog defaults, and either one kills a
# healthy server that is still compiling, as vLLM's do (see serve_vllm.sh).
export SGLANG_WARMUP_TIMEOUT="${SGLANG_WARMUP_TIMEOUT:-3600}"

ARGS=()
if [ $# -gt 0 ]; then
  case "$1" in
    --*)
      ARGS=("$@")
      ;;
    *)
      MODEL_PATH="$1"
      shift
      ARGS=(--model-path "$MODEL_PATH" "$@")
      ;;
  esac
fi

HAS_DEVICE=""
HAS_HOST=""
HAS_WATCHDOG=""
HAS_DIST_TIMEOUT=""
for arg in "${ARGS[@]}"; do
  case "$arg" in
    --device | --device=*) HAS_DEVICE=1 ;;
    --host | --host=*) HAS_HOST=1 ;;
    --watchdog-timeout | --watchdog-timeout=*) HAS_WATCHDOG=1 ;;
    --dist-timeout | --dist-timeout=*) HAS_DIST_TIMEOUT=1 ;;
  esac
done
[ -n "$HAS_DEVICE" ] || ARGS+=(--device tpu)
[ -n "$HAS_HOST" ] || ARGS+=(--host 0.0.0.0)

# The watchdog kills the server when a forward pass stalls, and collectives
# wait at most --dist-timeout, the same whether a rank is compiling or stuck.
# Only where this SGLang has the flag: an unknown one fails launch_server.
# One probe answers for both, so sglang.srt.server_args is imported once.
SERVER_ARG_FIELDS=" $("$PY" -c 'import sys
from sglang.srt.server_args import ServerArgs
print(" ".join(f for f in sys.argv[1:] if f in ServerArgs.__dataclass_fields__))' \
  watchdog_timeout dist_timeout 2>/dev/null || true) "
TIMEOUT_LOG="warmup=${SGLANG_WARMUP_TIMEOUT}s"
if [ -n "$HAS_WATCHDOG" ]; then
  TIMEOUT_LOG+=" watchdog=caller's"
elif [[ "$SERVER_ARG_FIELDS" == *" watchdog_timeout "* ]]; then
  ARGS+=(--watchdog-timeout "${SGLANG_WATCHDOG_TIMEOUT_S:-3600}")
  TIMEOUT_LOG+=" watchdog=${SGLANG_WATCHDOG_TIMEOUT_S:-3600}s"
else
  TIMEOUT_LOG+=" watchdog=left out (no --watchdog-timeout in this SGLang)"
fi
if [ -n "$HAS_DIST_TIMEOUT" ]; then
  TIMEOUT_LOG+=" dist=caller's"
elif [[ "$SERVER_ARG_FIELDS" == *" dist_timeout "* ]]; then
  ARGS+=(--dist-timeout "${SGLANG_DIST_TIMEOUT_S:-1800}")
  TIMEOUT_LOG+=" dist=${SGLANG_DIST_TIMEOUT_S:-1800}s"
else
  TIMEOUT_LOG+=" dist=left out (no --dist-timeout in this SGLang)"
fi
echo "serve_sglang: timeouts $TIMEOUT_LOG" >&2

echo "serve_sglang: profiler armed (SGLANG_TORCH_PROFILER_DIR=$PROFILE_DIR, PYTHON_TRACER_LEVEL=$PYTHON_TRACER_LEVEL)" >&2

FILES_LOG="${PROFILE_DIR%/}.files.log"
"$PY" -m http.server "$FILES_PORT" --directory "$PROFILE_DIR" \
  >"$FILES_LOG" 2>&1 &
FILES_PID=$!
sleep 1
if kill -0 "$FILES_PID" 2>/dev/null; then
  echo "serve_sglang: profile directory served on port $FILES_PORT" >&2
else
  echo "serve_sglang: WARNING: the file server on port $FILES_PORT did not" \
    "start ($(tail -n 1 "$FILES_LOG")); fetch traces with job artifacts" \
    "where the backend allows it" >&2
fi

"$PY" -m sglang.launch_server "${ARGS[@]}" &
SGLANG_PID=$!
trap 'kill "$SGLANG_PID" 2>/dev/null' TERM INT
set +e
wait "$SGLANG_PID"
STATUS=$?
while kill -0 "$SGLANG_PID" 2>/dev/null; do
  wait "$SGLANG_PID"
  STATUS=$?
done
kill "$FILES_PID" 2>/dev/null
exit "$STATUS"
