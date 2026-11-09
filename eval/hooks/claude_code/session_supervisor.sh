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

# ==============================================================================
# Claude Code Session Supervisor (/usr/local/bin/claude_session_supervisor.sh)
# ==============================================================================
# Shell script supervisor mounted inside evaluation containers.
# 1. Passes through informational commands (--version, help), and any claude
#    started from inside the session, directly.
# 1a. Runs the session as $AGENT_PERSONA (--agent), as the AGY supervisor does.
# 2. Removes Anthropic CLI's implicit 600s background task ceiling
#    (CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0) so multi-turn TPU agent suite iterations run
#    to completion without early termination.
# 3. Live-streams CLI execution while capturing raw event output.
# 4. Post-run: synthesizes the root session .jsonl under
#    $CLAUDE_CONFIG_DIR/projects/<slug>/<session-id>.jsonl and injects depth tags
#    [Depth <depth> | <agent>] across root and subagent sessions, mirroring AGY
#    supervisor behavior, so Harbor's native ATIF trajectory converter produces
#    a unified, fully attributed multi-agent trajectory.
# ==============================================================================
set -euo pipefail

SUPERVISOR_LOG="/logs/agent/session_supervisor.log"

log_supervisor() {
  local msg="$1"
  echo "$msg" >&2
  if [ -d "/logs/agent" ] || mkdir -p "/logs/agent" 2>/dev/null; then
    echo "$msg" >> "$SUPERVISOR_LOG" 2>/dev/null || true
  fi
}

# CLAUDE_REAL_BIN lets the tests stand in a fake CLI.
REAL_CLAUDE="${CLAUDE_REAL_BIN:-/usr/local/bin/claude-real}"
[ -x "$REAL_CLAUDE" ] || REAL_CLAUDE="/root/.local/bin/claude-real"
if [ ! -x "$REAL_CLAUDE" ]; then
  log_supervisor "[ClaudeSupervisor] Error: Native Claude CLI binary not found at $REAL_CLAUDE"
  exit 1
fi

# 1. Pass through version and inspection commands directly
case "${1:-}" in
  --version|-v|--help|help) exec "$REAL_CLAUDE" "$@" ;;
esac

# A `claude` started from inside the session (tpu-knowledge-base's summarising
# calls) is not the harness's session. Supervising it would run synthesis on
# exit, which rewrites the root session file while the root session writes it.
if [ -n "${CLAUDECODE:-}" ]; then
  exec "$REAL_CLAUDE" "$@"
fi

# A run that keeps no session is not one to synthesize. tpu-knowledge-base asks a
# nested `claude -p --no-session-persistence` from a temporary directory to
# summarise a page; a session made up for it would be a second project under
# the caller's CLAUDE_CONFIG_DIR, and Harbor will not pick a trajectory from
# two. Options end at `--`; what follows is the prompt.
for arg in "$@"; do
  [ "$arg" = "--" ] && break
  [ "$arg" = "--no-session-persistence" ] && exec "$REAL_CLAUDE" "$@"
done

# Run the session as the package persona, as the agy supervisor does, so both
# harnesses route through the same specialists. A task may override this with
# TASK_AGENT_PERSONA on packaged arms (where AGENT_PERSONA is set).
EFFECTIVE_PERSONA="${AGENT_PERSONA:-}"
if [ -n "$EFFECTIVE_PERSONA" ] && [ -n "${TASK_AGENT_PERSONA:-}" ]; then
  EFFECTIVE_PERSONA="$TASK_AGENT_PERSONA"
fi
CLI_ARGS=("$@")
if [ -n "$EFFECTIVE_PERSONA" ] && [[ ! " ${CLI_ARGS[*]} " =~ " --agent " ]]; then
  CLI_ARGS+=("--agent" "$EFFECTIVE_PERSONA")
fi

if ! command -v jq &>/dev/null; then
  log_supervisor "[ClaudeSupervisor] jq is not available; bypassing supervisor and executing native CLI directly."
  exec "$REAL_CLAUDE" "${CLI_ARGS[@]}"
fi

# 2. Environment overrides for non-interactive TPU agent suite and Multi-agent execution
export CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0
export IS_SANDBOX=1

# 3. Set up stream capture, session synthesis, and signal traps
RAW_STREAM="/tmp/claude_raw_stream_$$.txt"
CLAUDE_EXIT_CODE=0
STITCHED=0
CLAUDE_PID=""

# ==============================================================================
# Post-Run Session Synthesis & Depth Tagging
# ==============================================================================
synthesize_session() {
  trap '' INT TERM
  [ "${STITCHED:-0}" -eq 1 ] && return 0
  STITCHED=1
  [ -s "$RAW_STREAM" ] || return 0

  # Search all potential session directories (CLAUDE_CONFIG_DIR, /logs/agent/sessions, $HOME/.claude)
  local candidate_dirs=()
  for base in "${CLAUDE_CONFIG_DIR:-}" "/logs/agent/sessions" "${HOME}/.claude"; do
    if [ -n "$base" ] && [ -d "$base/projects" ]; then
      candidate_dirs+=("$base/projects")
    fi
  done

  local session_id
  session_id=$(grep -m1 -o '"session_id":"[^"]*"' "$RAW_STREAM" 2>/dev/null | cut -d'"' -f4)

  # The root session's project directory is the one holding its own
  # <session_id>.jsonl. Not the first directory listed: nested `claude -p`
  # calls (tpu-knowledge-base) add -tmp-* projects beside -workspace,
  # and synthesizing into one of those left two session directories, so Harbor
  # wrote no trajectory.json. Failing that, the directory Claude Code names
  # after the working directory.
  local proj_dir=""
  if [ -n "$session_id" ]; then
    for c in "${candidate_dirs[@]}"; do
      proj_dir=$(find "$c" -mindepth 2 -maxdepth 2 -type f -name "${session_id}.jsonl" ! -path '*/subagents/*' 2>/dev/null | head -n1)
      [ -n "$proj_dir" ] && proj_dir=$(dirname "$proj_dir") && break
    done
  fi
  if [ -z "$proj_dir" ]; then
    local default_base="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
    proj_dir="${default_base}/projects/$(pwd | sed 's/[^A-Za-z0-9]/-/g')"
  fi
  mkdir -p "$proj_dir"

  [ -z "$session_id" ] && session_id="session-$$"
  local target_jsonl="${proj_dir}/${session_id}.jsonl"

  # Helper jq filter definition to tag message text and thinking blocks idempotently
  local tag_fn='
    def tag_msg(t):
      if (.message?.content | type == "string") then
        if (.message.content | startswith("[Depth ")) then . else .message.content = (t + " " + .message.content) end
      elif (.message?.content | type == "array") then
        .message.content |= map(
          if (.type == "text" and (.text | type == "string") and (.text | startswith("[Depth ") | not)) then .text = (t + " " + .text)
          elif (.type == "thinking" and (.thinking | type == "string") and (.thinking | startswith("[Depth ") | not)) then .thinking = (t + " " + .thinking)
          else . end
        )
      else . end;'

  # 1. Tag all subagent transcripts in-place using jq (search recursively for all nested subagents)
  while IFS= read -r meta; do
    [ -f "$meta" ] || continue
    local agent_jsonl="${meta%.meta.json}.jsonl"
    [ -f "$agent_jsonl" ] || continue

    local agent_type
    agent_type=$(jq -r '.agentType // "subagent"' "$meta" 2>/dev/null || echo "subagent")
    local depth
    depth=$(jq -r '.spawnDepth // 1' "$meta" 2>/dev/null || echo 1)
    local tag="[Depth ${depth} | ${agent_type}]"

    if jq -R -c --arg tag "$tag" "${tag_fn} fromjson? | select(type == \"object\") | tag_msg(\$tag)" "$agent_jsonl" > "${agent_jsonl}.tmp" 2>/dev/null && [ -s "${agent_jsonl}.tmp" ]; then
      mv "${agent_jsonl}.tmp" "$agent_jsonl"
    else
      rm -f "${agent_jsonl}.tmp" 2>/dev/null || true
    fi
  done < <(find "$proj_dir" -type f -name "*.meta.json" 2>/dev/null)

  # 2. Tag root session file in-place if written by Claude, or synthesize from stream
  local root_tag="[Depth 0]"
  if [ -n "${EFFECTIVE_PERSONA:-}" ]; then
    root_tag="[Depth 0 | ${EFFECTIVE_PERSONA}]"
  fi
  if [ -s "$target_jsonl" ]; then
    log_supervisor "[ClaudeSupervisor] Tagging existing root session: $target_jsonl..."
    if jq -R -c --arg tag "$root_tag" "${tag_fn} fromjson? | select(type == \"object\") | tag_msg(\$tag)" "$target_jsonl" > "${target_jsonl}.tmp" 2>/dev/null && [ -s "${target_jsonl}.tmp" ]; then
      mv "${target_jsonl}.tmp" "$target_jsonl"
    else
      rm -f "${target_jsonl}.tmp" 2>/dev/null || true
    fi
  else
    log_supervisor "[ClaudeSupervisor] Synthesizing root session from stream: $RAW_STREAM -> $target_jsonl..."
    if jq -R -c --arg tag "$root_tag" "
      ${tag_fn}
      fromjson?
      | select(type == \"object\" and .type != null)
      | (if (.spawn_depth != null or .subagent_type != null) then
           \"[Depth \" + ((.spawn_depth // 1) | tostring) + \" | \" + (.subagent_type // \"subagent\") + \"]\"
         else \$tag end) as \$t
      | tag_msg(\$t)
    " "$RAW_STREAM" > "${target_jsonl}.tmp" 2>/dev/null && [ -s "${target_jsonl}.tmp" ]; then
      mv "${target_jsonl}.tmp" "$target_jsonl"
    else
      rm -f "${target_jsonl}.tmp" 2>/dev/null || true
    fi
  fi

  # 3. Fallback when CLI exited before emitting any user/assistant message event
  if [ ! -s "$target_jsonl" ] || ! jq -e 'select((.type == "assistant" or .type == "user") and (.message | type == "object"))' "$target_jsonl" >/dev/null 2>&1; then
    local fallback_txt
    fallback_txt=$(jq -R -s --arg tag "$root_tag" '
      (split("\n") | map(select(length > 0)) | map(
        (try fromjson catch null) as $obj
        | if ($obj | type) == "object" then
            ([$obj.result, $obj.error?.message, $obj.error, $obj.message]
             | map(select(type == "string" and length > 0)) | first // empty)
          else . end
      ) | join("\n")) as $body
      | if ($body | length) > 0 then ($tag + " [Early CLI Crash / Error Output]\n" + ($body | .[-4000:]))
        else ($tag + " [Early CLI Crash: empty stream]") end
    ' "$RAW_STREAM" 2>/dev/null || true)
    if [ -n "$fallback_txt" ]; then
      log_supervisor "[ClaudeSupervisor] Injecting fallback error step into $target_jsonl..."
      jq -nc --arg sid "$session_id" --argjson txt "$fallback_txt" \
        '{"type":"assistant","sessionId":$sid,"message":{"role":"assistant","content":[{"type":"text","text":$txt}]}}' \
        >> "$target_jsonl" 2>/dev/null || true
    fi
  fi

  log_supervisor "[ClaudeSupervisor] Session synthesis complete: $(wc -l < "$target_jsonl" 2>/dev/null || echo 0) root events."
}

# shellcheck disable=SC2317
cleanup() {
  trap '' INT TERM
  trap - EXIT
  if [ -n "${CLAUDE_PID:-}" ] && kill -0 "$CLAUDE_PID" 2>/dev/null; then
    pkill -TERM -P "$CLAUDE_PID" 2>/dev/null || true
    kill -TERM "$CLAUDE_PID" 2>/dev/null || true
    sleep 1
    if kill -0 "$CLAUDE_PID" 2>/dev/null; then
      kill -KILL "$CLAUDE_PID" 2>/dev/null || true
    fi
  fi
  wait 2>/dev/null || true
  synthesize_session || true
  rm -f "$RAW_STREAM" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 4. Launch real Claude CLI.
# Forward stdin (<&0), stream stdout+stderr live (so Harbor's tee receives output in real-time),
# and duplicate into RAW_STREAM for post-run session extraction and depth attribution.
# Background + wait allows SIGTERM/SIGINT (e.g. Harbor AgentTimeoutError) to interrupt wait
# immediately and run cleanup()/synthesize_session() before logs are downloaded.
log_supervisor "[ClaudeSupervisor] Launching native Claude Code: $REAL_CLAUDE ${CLI_ARGS[*]}"

set +e
exec 3> >(tee "$RAW_STREAM")
TEE_PID=$!
"$REAL_CLAUDE" "${CLI_ARGS[@]}" <&0 >&3 2>&3 3>&- &
CLAUDE_PID=$!
exec 3>&-
wait "$CLAUDE_PID" 2>/dev/null
CLAUDE_EXIT_CODE=$?
wait "$TEE_PID" 2>/dev/null || true
set -e

synthesize_session || true
sync || true

exit "$CLAUDE_EXIT_CODE"
