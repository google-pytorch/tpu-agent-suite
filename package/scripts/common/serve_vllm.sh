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
# Start `vllm serve` with its profiler armed, and serve the profile directory
# so a trace can be fetched while the server runs. Arguments go to `vllm serve`.
#
# vLLM >= 0.22 registers /start_profile only with --profiler-config, and
# vllm_torchtpu's worker writes TPU traces to its torch_profiler_dir (it no
# longer reads VLLM_TORCH_PROFILER_DIR, which is still set for older images).
# With our config the phased profiler is on (USE_PHASED_PROFILER=true,
# max_iterations=5): one short trace per inference phase rather than one long
# one. PYTHON_TRACER_LEVEL=0 keeps the Python tracer from filling the profiler's
# event buffer, which on a large model leaves host-only traces with no TPU
# timeline. None of it costs anything until /start_profile is called. The
# directory is served on its own port because `job artifacts` on gke_job
# cannot read a running pod's files.
#
# --xla_xprof_register_llo_debug_info puts the LLO source map in the trace,
# which `get_llo_analysis` and any kernel-level read need. libtpu reads
# LIBTPU_INIT_ARGS once, at backend start, so a server started without it
# cannot be given it later. On v7x (Qwen3-0.6B, libtpu 0.0.42.1) it made
# the trace 1.8x larger. Custom-call tracing is left off: it inflates traces
# 6-12x and skews kernel timings.
#
# Environment: VLLM_PROFILE_DIR (default artifacts/vllm_profile),
#              VLLM_PROFILE_FILES_PORT (default 8790),
#              VLLM_PROFILE_LLO (default 1; 0 leaves LIBTPU_INIT_ARGS as is),
#              VLLM_PROFILE_MAX_ITERATIONS (default 5; steps per phase),
#              USE_PHASED_PROFILER (default true when our --profiler-config is
#                used; a value the caller sets is kept),
#              PYTHON_TRACER_LEVEL (default 0; a value the caller sets is kept),
#              VLLM_ENGINE_READY_TIMEOUT_S (default 3600),
#              VLLM_ENGINE_ITERATION_TIMEOUT_S (default 1800),
#              VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS (default 1800),
#              VLLM_DISTRIBUTED_TIMEOUT_S (default 1800, passed as
#                --distributed-timeout-seconds when this vLLM has the flag);
#              a timeout the caller sets, or passes as a flag, is kept
# Patches: when the working directory holds vllm_patches.txt, each line names
# a patch file there, applied in order with apply_vllm_patch.sh (beside this
# script) before vLLM starts; one that does not apply (exit 4) fails the server.
# Usage, in launch_vllm.sh:
#   exec bash serve_vllm.sh "$(cat weights_path.txt)" --port 8000 ...

set -eo pipefail

PROFILE_DIR="${VLLM_PROFILE_DIR:-$PWD/artifacts/vllm_profile}"
FILES_PORT="${VLLM_PROFILE_FILES_PORT:-8790}"

# Resolved, since the default interpreter is a bare `python3`, and with a
# leading ~ expanded, since Kubernetes passes env values through as is.
PY="${TPU_AGENT_SUITE_PYTHON:-python3}"
PY="$(command -v "${PY/#\~/$HOME}" || true)"
VLLM="$(dirname "$PY")/vllm"
[ -x "$VLLM" ] || VLLM="$(command -v vllm)"

mkdir -p "$PROFILE_DIR"
export VLLM_TORCH_PROFILER_DIR="$PROFILE_DIR"
export PYTHON_TRACER_LEVEL="${PYTHON_TRACER_LEVEL:-0}"
# A value the caller gave the flag, true or false, is kept.
LLO_FLAG="--xla_xprof_register_llo_debug_info"
if [ "${VLLM_PROFILE_LLO:-1}" = 0 ]; then
  echo "serve_vllm: LLO debug info off (VLLM_PROFILE_LLO=0)" >&2
elif [[ " ${LIBTPU_INIT_ARGS:-} " == *" $LLO_FLAG"[=\ ]* ]]; then
  echo "serve_vllm: keeping the caller's $LLO_FLAG" >&2
else
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:+$LIBTPU_INIT_ARGS }$LLO_FLAG=true"
  echo "serve_vllm: LLO debug info on (LIBTPU_INIT_ARGS=$LIBTPU_INIT_ARGS)" >&2
fi

# vLLM's timeouts are sized for GPUs. Compiling a 30B model for TPU runs past
# the 600 s engine-ready, 60 s iteration and 300 s execute-model defaults, and
# a timeout there kills a healthy server that is still compiling: the most
# common failed first launch in onboarding runs (gemma-4-31B-it, Qwen3.8-27B).
export VLLM_ENGINE_READY_TIMEOUT_S="${VLLM_ENGINE_READY_TIMEOUT_S:-3600}"
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-1800}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
echo "serve_vllm: timeouts engine_ready=${VLLM_ENGINE_READY_TIMEOUT_S}s" \
  "iteration=${VLLM_ENGINE_ITERATION_TIMEOUT_S}s" \
  "execute_model=${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS}s" >&2

if [ -f vllm_patches.txt ]; then
  N=0
  while read -r name; do
    [ -n "$name" ] || continue
    rc=0
    bash "$(dirname "${BASH_SOURCE[0]}")/apply_vllm_patch.sh" "$name" >&2 </dev/null || rc=$?
    if [ "$rc" -eq 4 ]; then
      echo "serve_vllm: patch does not apply: $name" >&2
      exit 4
    elif [ "$rc" -ne 0 ]; then
      echo "serve_vllm: patch failed (exit $rc): $name" >&2
      exit "$rc"
    fi
    N=$((N + 1))
  done < vllm_patches.txt
  echo "serve_vllm: applied $N patch(es) from vllm_patches.txt" >&2
fi

# apply_vllm_patch.sh patches the plugin in place, which changes compiled
# programs but not the image the compile caches are keyed on: a patched build
# gets sibling folders of its own, named for the patches it has. They stay in
# the image's folder, so gke_job's compile-cache pruning removes them with it.
PATCHES="$("$PY" -c 'import importlib.util, os; s = importlib.util.find_spec("vllm_torchtpu"); print("VLLM_TORCHTPU_PKG=" + list(s.submodule_search_locations)[0])' 2>/dev/null \
  | sed -n 's/^VLLM_TORCHTPU_PKG=//p' | tail -n 1 || true)/.tpu_agent_suite_patches"
if [ -s "$PATCHES" ]; then
  KEY="patch-$(sort -u "$PATCHES" | sha256sum | cut -c1-12)"
  for var in VLLM_CACHE_ROOT TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT \
    TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT; do
    if [ -n "${!var:-}" ]; then
      export "$var=${!var%/}-$KEY"
    fi
  done
  echo "serve_vllm: patched vllm_torchtpu; compile caches under *-$KEY" >&2
fi

ARGS=("$@")
OWN_CONFIG=""
for arg in "$@"; do
  case "$arg" in
    --profiler-config | --profiler-config[=.]*) OWN_CONFIG=1 ;;
  esac
done
# Appending ours after the caller's would silently override it.
if [ -n "$OWN_CONFIG" ]; then
  # Only ours is known to carry torch_profiler_dir, without which the phased
  # profiler refuses to start, so it is left as the caller set it.
  echo "serve_vllm: keeping the caller's --profiler-config; TPU traces go" \
    "to its torch_profiler_dir, and USE_PHASED_PROFILER is left as set" >&2
elif "$PY" -c "import vllm.config.profiler" >/dev/null 2>&1; then
  # The module ships with --profiler-config; cheaper than `--help=all`.
  export USE_PHASED_PROFILER="${USE_PHASED_PROFILER:-true}"
  if [ "$USE_PHASED_PROFILER" = true ] || [ -n "${VLLM_PROFILE_MAX_ITERATIONS:-}" ]; then
    MAX_ITERS="${VLLM_PROFILE_MAX_ITERATIONS:-5}"
    ARGS+=(--profiler-config
      "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$PROFILE_DIR\", \"max_iterations\": $MAX_ITERS, \"ignore_frontend\": true}")
  else
    MAX_ITERS="none"
    ARGS+=(--profiler-config
      "{\"profiler\": \"torch\", \"torch_profiler_dir\": \"$PROFILE_DIR\", \"ignore_frontend\": true}")
  fi
  echo "serve_vllm: profiler armed (--profiler-config," \
    "USE_PHASED_PROFILER=$USE_PHASED_PROFILER," \
    "PYTHON_TRACER_LEVEL=$PYTHON_TRACER_LEVEL," \
    "max_iterations=$MAX_ITERS), traces under $PROFILE_DIR" >&2
else
  echo "serve_vllm: WARNING: this vLLM has no --profiler-config; only" \
    "VLLM_TORCH_PROFILER_DIR is set. If /openapi.json lists no" \
    "/start_profile, this server cannot be profiled." >&2
fi

# Collectives wait at most this long, the same whether a rank is compiling or
# stuck. Only where this vLLM has the flag: an unknown one fails `vllm serve`.
OWN_DIST_TIMEOUT=""
for arg in "$@"; do
  case "$arg" in
    --distributed-timeout-seconds | --distributed-timeout-seconds=*) OWN_DIST_TIMEOUT=1 ;;
  esac
done
if [ -z "$OWN_DIST_TIMEOUT" ] && "$PY" -c 'import sys
from vllm.engine.arg_utils import EngineArgs
sys.exit("distributed_timeout_seconds" not in EngineArgs.__dataclass_fields__)' \
  >/dev/null 2>&1; then
  ARGS+=(--distributed-timeout-seconds "${VLLM_DISTRIBUTED_TIMEOUT_S:-1800}")
fi

FILES_LOG="${PROFILE_DIR%/}.files.log"
# No --bind: dual-stack, so the port-forward reaches it as it reaches vLLM.
"$PY" -m http.server "$FILES_PORT" --directory "$PROFILE_DIR" \
  >"$FILES_LOG" 2>&1 &
FILES_PID=$!
sleep 1
if kill -0 "$FILES_PID" 2>/dev/null; then
  echo "serve_vllm: profile directory served on port $FILES_PORT" >&2
else
  echo "serve_vllm: WARNING: the file server on port $FILES_PORT did not" \
    "start ($(tail -n 1 "$FILES_LOG")); fetch traces with job artifacts" \
    "where the backend allows it" >&2
fi

"$VLLM" serve "${ARGS[@]}" &
VLLM_PID=$!
# Pass a cancel on to vLLM, and never let the file server outlive it.
trap 'kill "$VLLM_PID" 2>/dev/null' TERM INT
set +e
wait "$VLLM_PID"
STATUS=$?
# A trapped signal ends the first wait early.
while kill -0 "$VLLM_PID" 2>/dev/null; do
  wait "$VLLM_PID"
  STATUS=$?
done
kill "$FILES_PID" 2>/dev/null
exit "$STATUS"
