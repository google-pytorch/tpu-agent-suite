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
# Antigravity CLI Session Supervisor (/root/.local/bin/agy)
# ==============================================================================
# Pure Shell script supervisor mounted inside evaluation containers.
# Maintains open stdin for streaming multi-agent execution, translates CLI prompts
# into streaming JSON events, and detects subagent quiescence before exiting.
set -euo pipefail

REAL_AGY="${AGY_REAL_BIN:-/usr/local/bin/agy-real}"
[ -x "$REAL_AGY" ] || REAL_AGY="/root/.local/bin/agy-real"
export AGY_ADC_AUTH="${AGY_ADC_AUTH:-1}"
# The eval pins the CLI's version; a background self-update would replace it.
export AGY_CLI_DISABLE_AUTO_UPDATE=true
# Only when an ADC file is mounted: pointing this at a missing file makes
# ADC fail instead of falling back to the pod's GKE metadata server.
if [ -z "${GOOGLE_APPLICATION_CREDENTIALS:-}" ] && [ -f /root/.config/gcloud/application_default_credentials.json ]; then
  export GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json
fi
export GOOGLE_CLOUD_QUOTA_PROJECT="${GOOGLE_CLOUD_QUOTA_PROJECT:-${VERTEXAI_PROJECT:-tpu-nexus-767207}}"

SUPERVISOR_LOG="/logs/agent/session_supervisor.log"

log_supervisor() {
  local msg="$1"
  echo "$msg" >&2
  if [ -d "/logs/agent" ] || mkdir -p "/logs/agent" 2>/dev/null; then
    echo "$msg" >> "$SUPERVISOR_LOG" 2>/dev/null || true
  fi
}

if [ ! -x "$REAL_AGY" ]; then
  log_supervisor "[AgySupervisor] Error: Native Antigravity CLI binary not found at $REAL_AGY"
  exit 1
fi

# 1. Pass through inspection commands directly
if [[ "$1" == "--version" || "$1" == "models" || "$1" == "agents" ]]; then
  exec "$REAL_AGY" "$@"
fi

if ! command -v jq &>/dev/null; then
  log_supervisor "[AgySupervisor] jq is not available; bypassing supervisor and executing native CLI directly."
  exec "$REAL_AGY" "$@"
fi

# 2. Extract prompt argument; forward all other flags to agy
PROMPT=""
CLI_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --prompt=*)
      PROMPT="${1#--prompt=}"
      shift
      ;;
    --prompt)
      PROMPT="$2"
      shift 2
      ;;
    *)
      CLI_ARGS+=("$1")
      shift
      ;;
  esac
done

EFFECTIVE_PERSONA="${AGENT_PERSONA:-}"
if [ -n "$EFFECTIVE_PERSONA" ] && [ -n "${TASK_AGENT_PERSONA:-}" ]; then
  EFFECTIVE_PERSONA="$TASK_AGENT_PERSONA"
fi
CLI_ARGS+=("--input-format" "stream-json")
if [ -n "$EFFECTIVE_PERSONA" ] && [[ ! " ${CLI_ARGS[*]} " =~ " --agent " ]]; then
  CLI_ARGS+=("--agent" "$EFFECTIVE_PERSONA")
fi

log_supervisor "[AgySupervisor] Launching native Antigravity CLI: $REAL_AGY ${CLI_ARGS[*]}"

# 3. Output stream capture and FIFO state
RAW_STREAM="/tmp/agy_raw_stream_$$.jsonl"
FIFO="/tmp/agy_stdin_$$"
AGY_PID=""
STITCHED=0

# ==============================================================================
# Post-Run Trajectory Stitcher (100% Pure Shell via jq)
# ==============================================================================
# Harbor collects a single trajectory from ~/.gemini/antigravity-cli/brain via:
#   src=$(find "$HOME/.gemini/antigravity-cli/brain" -type f -name 'transcript_full.jsonl' ... | sort -nr | head -n1)
# Antigravity CLI isolates child subagents in separate UUID directories.
# To show the complete end-to-end multi-agent execution in Harbor UI, we stitch
# all subagent transcripts chronologically into a unified trajectory file, ensure
# it has the newest timestamp, and sync it across transcript files.
# shellcheck disable=SC2317
stage_harbor_trajectory() {
  local src="$1"
  local logs_dir="${LOGS_AGENT_DIR:-/logs/agent}"
  if mkdir -p "$logs_dir" 2>/dev/null; then
    local tmp="${logs_dir}/.antigravity-cli.trajectory.jsonl.$$.tmp"
    if cp "$src" "$tmp" 2>/dev/null; then
      mv -f "$tmp" "${logs_dir}/antigravity-cli.trajectory.jsonl" 2>/dev/null || rm -f "$tmp" 2>/dev/null || true
    else
      rm -f "$tmp" 2>/dev/null || true
    fi
  fi
}

# Aggregate token usage across root and all child subagent conversations.
# Antigravity CLI stream-json output only reports usage for the Depth 0 root
# conversation, while writing per-turn ModelUsageStats protobuf records into
# ~/.gemini/antigravity-cli/conversations/<id>.db (table gen_metadata) for
# both the root session and every invoked subagent.
# shellcheck disable=SC2317
aggregate_conversation_usage() {
  command -v python3 &>/dev/null || return 0
  local logs_dir="${LOGS_AGENT_DIR:-/logs/agent}"
  local stream_target="${logs_dir}/antigravity-stream.jsonl"
  local agg_line
  agg_line=$(python3 - "${HOME}/.gemini/antigravity-cli/conversations" "$RAW_STREAM" "$stream_target" "${HARBOR_RUN_MARKER:-/tmp/.harbor-antigravity-run-start}" <<'PY' 2>/dev/null || true
import glob, json, os, sqlite3, sys

KEYS = ("input_tokens", "output_tokens", "thinking_tokens", "cache_read_tokens")
FIELDS = {2: "input_tokens", 3: "output_tokens", 5: "cache_read_tokens", 9: "thinking_tokens"}

def _varint(b: bytes, i: int) -> tuple[int, int]:
    v = s = 0
    while b[i] >= 0x80:
        v |= (b[i] & 0x7F) << s
        s += 7
        i += 1
    return v | (b[i] << s), i + 1

def _fields(b: bytes):
    i, n = 0, len(b)
    while i < n:
        tag, i = _varint(b, i)
        f, w = tag >> 3, tag & 7
        if w == 0:
            v, i = _varint(b, i)
        elif w == 2:
            ln, i = _varint(b, i)
            v, i = b[i : i + ln], i + ln
        elif w in (1, 5):
            step = 8 if w == 1 else 4
            v, i = b[i : i + step], i + step
        else:
            break
        yield f, w, v

def _parse_blob(blob: bytes) -> dict[str, int]:
    u = {k: 0 for k in KEYS}
    for f, w, v in _fields(blob):
        if f == 1 and w == 2:  # CortexStepGeneratorMetadata.chat_model
            for cf, cw, cv in _fields(v):
                if cf == 4 and cw == 2:  # ChatModelMetadata.usage (ModelUsageStats)
                    for uf, uw, uv in _fields(cv):
                        if uw == 0 and uf in FIELDS:
                            u[FIELDS[uf]] = int(uv)
    return u

def _norm(d: dict) -> dict[str, int]:
    return {
        k: int(d[k]) if isinstance(d.get(k), (int, float)) and not isinstance(d.get(k), bool) else 0
        for k in KEYS
    }

def _max(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    return {k: max(a[k], b[k]) for k in KEYS}

conv_dir, raw_stream, stream_file, run_marker = sys.argv[1:5]
min_mtime = (os.path.getmtime(run_marker) - 1.0) if run_marker and os.path.exists(run_marker) else None

root_id, status, response, res_usage = "", "DONE", "", None
steps: dict[int, dict[str, int]] = {}
for path in (raw_stream, stream_file):
    if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
        continue
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if not isinstance(ev, dict):
                continue
            root_id = root_id or str(
                ev.get("conversation_id")
                or ev.get("session_id")
                or (ev.get("init") or {}).get("conversation_id")
                or (ev.get("session_start") or {}).get("conversation_id")
                or ""
            )
            if ev.get("event") == "step_update":
                su = ev.get("step_update") or {}
                root_id = root_id or str(su.get("conversation_id") or "")
                if isinstance(su.get("usage"), dict):
                    idx = su.get("step_index")
                    key = idx if isinstance(idx, int) and not isinstance(idx, bool) else len(steps) + 1_000_000
                    steps[key] = _max(steps.get(key, {k: 0 for k in KEYS}), _norm(su["usage"]))
            elif ev.get("event") == "result":
                res = ev.get("result") or {}
                root_id = root_id or str(res.get("conversation_id") or "")
                status = str(res.get("status") or status)
                response = str(res.get("response") or response)
                if isinstance(res.get("usage"), dict):
                    res_usage = _max(res_usage or {k: 0 for k in KEYS}, _norm(res["usage"]))
    break

stream_usage = res_usage or {k: sum(u[k] for u in steps.values()) for k in KEYS}

db_usages: dict[str, dict[str, int]] = {}
for db_path in sorted(glob.glob(os.path.join(conv_dir, "*.db"))):
    if min_mtime is not None and os.path.getmtime(db_path) < min_mtime:
        continue
    try:
        with sqlite3.connect(db_path, timeout=2.0) as conn:
            rows = conn.execute("SELECT data FROM gen_metadata ORDER BY idx").fetchall()
    except Exception:
        continue
    u = {k: 0 for k in KEYS}
    for (blob,) in rows:
        if blob:
            try:
                turn = _parse_blob(bytes(blob))
                for k in KEYS:
                    u[k] += turn[k]
            except Exception:
                pass
    db_usages[os.path.splitext(os.path.basename(db_path))[0]] = u

if not root_id and db_usages:
    root_id = next(iter(db_usages))
root_usage = _max(db_usages.pop(root_id, {k: 0 for k in KEYS}), stream_usage)
total = {k: root_usage[k] + sum(u[k] for u in db_usages.values()) for k in KEYS}

if any(total.values()):
    total["total_tokens"] = total["input_tokens"] + total["output_tokens"]
    print(json.dumps({
        "event": "result",
        "session_id": root_id,
        "result": {"conversation_id": root_id, "status": status, "response": response, "usage": total},
    }, separators=(",", ":")))
PY
  )
  if [ -n "$agg_line" ]; then
    printf '%s\n' "$agg_line" 2>/dev/null || true
    if [ -d "$logs_dir" ] || mkdir -p "$logs_dir" 2>/dev/null; then
      printf '%s\n' "$agg_line" >> "$stream_target" 2>/dev/null || true
    fi
  fi
}

# shellcheck disable=SC2317
synthesize_fallback_agy_trajectory() {
  local brain_dir="${HOME}/.gemini/antigravity-cli/brain"
  local stream_file=""
  for candidate in "$RAW_STREAM" "/logs/agent/antigravity-stream.jsonl"; do
    if [ -s "$candidate" ]; then
      stream_file="$candidate"
      break
    fi
  done
  local err_file="/logs/agent/antigravity-cli.err"
  if [ -z "$PROMPT" ] && [ -z "$stream_file" ] && [ ! -s "$err_file" ]; then
    return 0
  fi

  local root_tag="[Depth 0]"
  if [ -n "${EFFECTIVE_PERSONA:-}" ]; then
    root_tag="[Depth 0 | ${EFFECTIVE_PERSONA}]"
  fi

  local now_ts
  now_ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || echo "1970-01-01T00:00:00Z")
  local err_tail=""
  if [ -s "$err_file" ]; then
    err_tail=$(tail -c 4000 "$err_file" 2>/dev/null || true)
  fi
  local stream_err=""
  if [ -n "$stream_file" ]; then
    stream_err=$(jq -R -r '
      fromjson? | select(type == "object")
      | ([.result?.error, .result?.response, .error?.message, .error]
         | map(select(type == "string" and length > 0)) | first // empty)
    ' "$stream_file" 2>/dev/null | tail -c 4000 || true)
  fi

  local combined_err="${stream_err}"
  if [ -n "$err_tail" ]; then
    if [ -n "$combined_err" ]; then
      combined_err="${combined_err}"$'\n'"${err_tail}"
    else
      combined_err="${err_tail}"
    fi
  fi
  [ -z "$combined_err" ] && combined_err="Antigravity CLI exited before writing a native transcript."

  local unified_dir="${brain_dir}/unified/.system_generated/logs"
  mkdir -p "$unified_dir" 2>/dev/null || true
  local unified_file="${unified_dir}/transcript_full.jsonl"
  local prompt_text="${PROMPT:-[Early CLI invocation]}"

  {
    jq -nc \
      --arg ts "$now_ts" \
      --arg content "${root_tag} ${prompt_text}" \
      '{"step_index":0,"type":"USER_INPUT","source":"USER_EXPLICIT","status":"DONE","created_at":$ts,"content":$content}'
    jq -nc \
      --arg ts "$now_ts" \
      --arg content "${root_tag} [Early CLI Crash / Error Output]"$'\n'"${combined_err}" \
      --arg err "$combined_err" \
      '{"step_index":1,"type":"SYSTEM_ERROR","source":"SYSTEM","status":"ERROR","created_at":$ts,"content":$content,"error":$err}'
  } > "${unified_file}.tmp" 2>/dev/null || true

  if [ -s "${unified_file}.tmp" ]; then
    mv "${unified_file}.tmp" "$unified_file"
    touch "$unified_file" 2>/dev/null || true
    if [ -d "$brain_dir" ]; then
      local existing_f
      while IFS= read -r existing_f; do
        [ -f "$existing_f" ] || continue
        cp "$unified_file" "$existing_f" 2>/dev/null || true
        local compact="${existing_f%/transcript_full.jsonl}/transcript.jsonl"
        if [ -f "$compact" ]; then
          cp "$unified_file" "$compact" 2>/dev/null || true
        fi
      done < <(find "$brain_dir" -type f -name 'transcript_full.jsonl' ! -path '*/unified/*' 2>/dev/null)
    fi
    stage_harbor_trajectory "$unified_file"
    log_supervisor "[AgySupervisor] Synthesized fallback trajectory at $unified_file from stream/error logs."
  else
    rm -f "${unified_file}.tmp" 2>/dev/null || true
  fi
}

# shellcheck disable=SC2317
stitch_trajectories() {
  if [ "$STITCHED" -eq 1 ]; then
    return 0
  fi
  STITCHED=1

  local brain_dir="${HOME}/.gemini/antigravity-cli/brain"
  local files=()
  if [ -d "$brain_dir" ]; then
    while IFS= read -r f; do
      [ -f "$f" ] && files+=("$f")
    done < <(find "$brain_dir" -type f -name 'transcript_full.jsonl' ! -path '*/unified/*' 2>/dev/null)
  fi

  if [ "${#files[@]}" -eq 0 ]; then
    synthesize_fallback_agy_trajectory || true
    aggregate_conversation_usage || true
    return 0
  fi

  # Discover subagent conversation metadata from the stream logs or transcripts
  declare -A conv_meta=()
  local stream
  for stream in "$RAW_STREAM" "/logs/agent/antigravity-stream.jsonl"; do
    if [ -f "$stream" ]; then
      while IFS=$'\t' read -r cid tname; do
        if [ -n "$cid" ] && [ -n "$tname" ]; then
          conv_meta["$cid"]="${tname}|1"
        fi
      done < <(grep '"subagent_info"' "$stream" 2>/dev/null | jq -r '
        .step_update?.subagent_info?.subagents[]?
        | [.conversation_id, (.type_name // .role // "subagent")]
        | @tsv
      ' 2>/dev/null || true)
    fi
  done

  # Also inspect transcripts for conversationId references
  for f in "${files[@]}"; do
    while IFS= read -r cid; do
      if [ -n "$cid" ] && [ -z "${conv_meta[$cid]:-}" ]; then
        conv_meta["$cid"]="subagent|1"
      fi
    done < <(grep -o '"conversationId": *"[^"]*"' "$f" 2>/dev/null | cut -d'"' -f4 || true)
  done

  local db_file="${HOME}/.gemini/antigravity-cli/conversation_summaries.db"

  local tagged_files=()

  # Tag each conversation with its agent identity and depth
  for f in "${files[@]}"; do
    local conv_id
    conv_id=$(basename "$(dirname "$(dirname "$(dirname "$f")")")")
    local agent_name=""
    local depth=0

    if [ -f "$db_file" ]; then
      local meta
      meta=$(sqlite3 -cmd ".timeout 2000" "$db_file" "SELECT COALESCE(NULLIF(agent_name, ''), NULLIF('${EFFECTIVE_PERSONA:-}', '')) || '|' || COALESCE(nesting_depth, 0) FROM conversation_summaries WHERE conversation_id = '$conv_id';" 2>/dev/null || true)
      if [ -n "$meta" ]; then
        agent_name="${meta%|*}"
        depth="${meta##*|}"
      fi
    fi
    if [ -z "$agent_name" ] && [ -n "${conv_meta[$conv_id]:-}" ]; then
      local meta="${conv_meta[$conv_id]}"
      agent_name="${meta%|*}"
      depth="${meta##*|}"
    elif [ -z "$agent_name" ]; then
      agent_name="${EFFECTIVE_PERSONA:-}"
      depth=0
    fi
    [ -z "$depth" ] && depth=0

    local tag="[Depth ${depth}]"
    if [ "$depth" -eq 0 ]; then
      if [ -n "$agent_name" ]; then
        tag="[Depth 0 | ${agent_name}]"
      fi
    else
      local display_name="${agent_name:-subagent}"
      tag="[Depth ${depth} | ${display_name}]"
    fi

    local tagged_tmp="${f}.tagged"
    jq -R -c --arg agent "${agent_name:-agent}" --argjson depth "$depth" --arg tag "$tag" '
      fromjson?
      | select(type == "object" and .created_at != null and .type != null)
      | .agent = $agent
      | .depth = $depth
      | if (.content | type == "string" and length > 0) then
          (if (.content | startswith("[Depth ")) then . else .content = ($tag + " " + .content) end)
        elif (.tool_calls != null or .thinking != null) then
          .content = $tag
        else
          .
        end
      | if (.thinking | type == "string" and length > 0) then
          (if (.thinking | startswith("[Depth ")) then . else .thinking = ($tag + " " + .thinking) end)
        else
          .
        end
    ' "$f" > "$tagged_tmp" 2>/dev/null || true
    tagged_files+=("$tagged_tmp")
  done

  # If single conversation (e.g. vanilla AGY baseline), overwrite in-place
  if [ "${#files[@]}" -eq 1 ]; then
    local single_file="${files[0]}"
    local tagged_single="${tagged_files[0]}"
    if [ -s "$tagged_single" ] && jq -e 'select(.type == "USER_INPUT" or .type == "PLANNER_RESPONSE")' "$tagged_single" >/dev/null 2>&1; then
      mv "$tagged_single" "$single_file"
      local compact="${single_file%/transcript_full.jsonl}/transcript.jsonl"
      if [ -f "$compact" ]; then
        cp "$single_file" "$compact" 2>/dev/null || true
      fi
      stage_harbor_trajectory "$single_file"
      log_supervisor "[AgySupervisor] Tagged single session trajectory with [Depth 0]."
    else
      rm -f "$tagged_single" 2>/dev/null || true
      synthesize_fallback_agy_trajectory || true
    fi
    aggregate_conversation_usage || true
    return 0
  fi

  log_supervisor "[AgySupervisor] Found ${#files[@]} agent transcripts. Stitching into unified trajectory via jq..."

  local unified_dir="${brain_dir}/unified/.system_generated/logs"
  mkdir -p "$unified_dir"
  local unified_file="${unified_dir}/transcript_full.jsonl"

  # Merge all tagged records, sort chronologically by created_at, and renumber step_index 0..N-1
  cat "${tagged_files[@]}" | jq -c -s '
    sort_by(.created_at)
    | to_entries
    | map(.value + {step_index: .key})[]
  ' > "$unified_file.tmp" 2>/dev/null || true

  rm -f "${tagged_files[@]}" 2>/dev/null || true

  if [ -s "$unified_file.tmp" ] && jq -e 'select(.type == "USER_INPUT" or .type == "PLANNER_RESPONSE")' "$unified_file.tmp" >/dev/null 2>&1; then
    mv "$unified_file.tmp" "$unified_file"
    touch "$unified_file"
    for f in "${files[@]}"; do
      cp "$unified_file" "$f" 2>/dev/null || true
      local compact="${f%/transcript_full.jsonl}/transcript.jsonl"
      if [ -f "$compact" ]; then
        cp "$unified_file" "$compact" 2>/dev/null || true
      fi
    done
    stage_harbor_trajectory "$unified_file"
    log_supervisor "[AgySupervisor] Stitched unified trajectory generated: $(wc -l < "$unified_file") steps with agent tags."
  else
    rm -f "$unified_file.tmp"
    synthesize_fallback_agy_trajectory || true
  fi
  aggregate_conversation_usage || true
}

# Cleanup trap to ensure FIFO, background process, and trajectory stitching are handled
# shellcheck disable=SC2317
cleanup() {
  trap '' INT TERM PIPE
  trap - EXIT
  exec 3>&- 2>/dev/null || true
  rm -f "$FIFO" 2>/dev/null || true
  if [ -n "${AGY_PID:-}" ] && kill -0 "$AGY_PID" 2>/dev/null; then
    pkill -TERM -P "$AGY_PID" 2>/dev/null || true
    kill -TERM "$AGY_PID" 2>/dev/null || true
    sleep 1
    if kill -0 "$AGY_PID" 2>/dev/null; then
      pkill -KILL -P "$AGY_PID" 2>/dev/null || true
      kill -KILL "$AGY_PID" 2>/dev/null || true
    fi
  fi
  wait 2>/dev/null || true
  stitch_trajectories || true
  rm -f "$RAW_STREAM" 2>/dev/null || true
}
trap 'cleanup; exit 143' INT TERM
trap cleanup EXIT

# 4. Launch agy-real with streaming I/O bound to the FIFO and capture stream output
mkfifo "$FIFO"
exec 3<>"$FIFO"
"$REAL_AGY" "${CLI_ARGS[@]}" <"$FIFO" 3>&- > >(tee "$RAW_STREAM") &
AGY_PID=$!

# 5. Dispatch initial prompt event over the open pipe (RFC 8259 JSON via jq)
if [ -n "$PROMPT" ]; then
  jq -nc --arg p "$PROMPT" '{"event":"user","message":{"content":$p}}' >&3
fi

# 6. Active Agent Completion Checker
check_work_completed() {
  # 1. Primary check: Native stream event completion in RAW_STREAM or /logs/agent/antigravity-stream.jsonl
  local stream
  for stream in "$RAW_STREAM" "/logs/agent/antigravity-stream.jsonl"; do
    if [ -f "$stream" ] && grep -q '"event":"result"' "$stream" 2>/dev/null; then
      return 0
    fi
  done

  # 2. Fallback check: conversation_summaries.db (e.g. AGY full desktop/daemon)
  local db_file="${HOME}/.gemini/antigravity-cli/conversation_summaries.db"
  if [ -f "$db_file" ]; then
    local total
    total=$(sqlite3 -cmd ".timeout 2000" "$db_file" "SELECT COUNT(*) FROM conversation_summaries;" 2>/dev/null || echo "0")
    if [ "${total:-0}" -gt 0 ]; then
      local active
      active=$(sqlite3 -cmd ".timeout 2000" "$db_file" "SELECT COUNT(*) FROM conversation_summaries WHERE status LIKE '%RUNNING%' OR not_fully_idle != 0;" 2>/dev/null || echo "1")
      if [ "${active:-1}" -eq 0 ]; then
        return 0
      fi
    fi
  fi

  return 1
}

# 7. Monitoring Loop: Keep looking for active working agents to complete their work
WORK_DONE=0
while kill -0 "$AGY_PID" 2>/dev/null; do
  sleep 2
  if check_work_completed; then
    log_supervisor "[AgySupervisor] All working agents completed their work. Concluding session."
    WORK_DONE=1
    break
  fi
done

# 8. Close stdin to signal EOF to agy-real and wait for clean exit
exec 3>&- 2>/dev/null || true
rm -f "$FIFO" 2>/dev/null || true

set +e
WAIT_COUNT=0
while kill -0 "$AGY_PID" 2>/dev/null && [ "$WAIT_COUNT" -lt 15 ]; do
  sleep 1
  WAIT_COUNT=$((WAIT_COUNT + 1))
done

if kill -0 "$AGY_PID" 2>/dev/null; then
  kill -TERM "$AGY_PID" 2>/dev/null || true
  sleep 2
  if kill -0 "$AGY_PID" 2>/dev/null; then
    kill -KILL "$AGY_PID" 2>/dev/null || true
  fi
fi

AGY_EXIT_CODE=0
wait "$AGY_PID" 2>/dev/null || AGY_EXIT_CODE=$?
wait 2>/dev/null || true

stitch_trajectories || true

if [ "$WORK_DONE" -eq 1 ]; then
  exit 0
fi
exit "$AGY_EXIT_CODE"
