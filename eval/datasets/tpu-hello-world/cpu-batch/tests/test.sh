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

# Three aspects, and the third is the one this task exists for.
#
# The numbers are the easy part. What is actually under test is whether the
# agent asked the user for a batch size instead of choosing one -- and that
# cannot be read off the result table, because a table reporting `8` looks
# identical whether the user said 8 or the agent liked the number. It has to be
# read from the session: did an ask reach the user, and is what ran what came
# back.
set -euo pipefail

LOGS="${LOGS_DIR:-/logs/verifier}"
mkdir -p "$LOGS"

AGENT_LOGS=("${AGENT_LOGS_DIR:-/logs/agent}"/*)
# A simulated-user trial puts the conversation somewhere else. The coding
# agent is reached over a bridge, so what it said -- the result table
# included -- is relayed through the user agent and lands in its log, while
# `/logs/agent` holds only the bridge's protocol dump. Reading just the one
# directory scores a bridged run as though the agent had said nothing.
USER_AGENT_DIR="${USER_AGENT_LOGS_DIR:-/logs/user-agent}"
if [ -d "$USER_AGENT_DIR" ]; then
  AGENT_LOGS+=("$USER_AGENT_DIR"/*)
fi

# A whole cell: a non-empty value and the pipe that closes it. The bridge
# dump carries the same table split across streamed message chunks, so a row
# can end mid-value (`| device_type | cpu`), and the last match wins. Without
# the closing pipe that fragment beat the well-formed rows in the agent's own
# logs; requiring a value as well as the pipe skips the empty template the
# task instruction carries.
table_row() {
  { grep -ohE "\|[[:space:]]*$1[[:space:]]*\|[[:space:]]*[^|[:space:]][^|]*\|" \
      "${AGENT_LOGS[@]}" 2>/dev/null || true; } | tail -n 1 |
    sed -E "s/.*\|[[:space:]]*$1[[:space:]]*\|[[:space:]]*//; s/\|\$//" |
    tr -d '[:space:]'
}

DEVICE_TYPE="$(table_row device_type)"
BATCH_MODE="$(table_row batch_mode)"
BATCH_SIZES="$(table_row batch_sizes)"
MAX_ABS_DIFF="$(table_row max_abs_diff)"
TOLERANCE="$(table_row tolerance)"

# The scored artifact. A file the agent writes is one unambiguous source, so
# it wins over anything read back out of the logs: the table reaches them as
# prose, split across streamed message chunks and repeated by every component
# that echoes it. The table stays the human-facing report and the fallback for
# a run that did not write the file.
RESULT_JSON="${RESULT_JSON:-/logs/artifacts/result.json}"

# Over a bridge, read the result table out of the agent's last message that
# carries one. Grepping each field from the logs independently takes the last
# match per field, so a run that also prints a per-size table answers from two
# different tables at once. The task says the final response starts with the
# table, and the conversation is where that message is.
BRIDGE="${AGENT_LOGS_DIR:-/logs/agent}/bridge-trajectory.json"
if [ -f "$BRIDGE" ]; then
  while IFS='=' read -r key value; do
    case "$key" in
    device_type) DEVICE_TYPE="$value" ;;
    batch_mode) BATCH_MODE="$value" ;;
    batch_sizes) BATCH_SIZES="$value" ;;
    max_abs_diff) MAX_ABS_DIFF="$value" ;;
    tolerance) TOLERANCE="$value" ;;
    esac
  done < <(python3 "$(dirname "$0")/read_bridge.py" table "$BRIDGE" || true)
fi

if [ -f "$RESULT_JSON" ]; then
  while IFS='=' read -r key value; do
    case "$key" in
    device_type) DEVICE_TYPE="$value" ;;
    batch_mode) BATCH_MODE="$value" ;;
    batch_sizes) BATCH_SIZES="$value" ;;
    max_abs_diff) MAX_ABS_DIFF="$value" ;;
    tolerance) TOLERANCE="$value" ;;
    esac
  done < <(python3 -c '
import json, sys
try:
    data = json.load(open(sys.argv[1], encoding="utf-8"))
except Exception:
    sys.exit(0)          # unreadable: the table still scores the run
if isinstance(data, dict):
    for key in ("device_type", "batch_mode", "batch_sizes", "max_abs_diff",
                "tolerance"):
        value = data.get(key)
        if isinstance(value, (str, int, float)):
            print(f"{key}={str(value).strip()}")
' "$RESULT_JSON" || true)
fi

# Aspect 1: the CPU path, taken deliberately.
DEVICE_OK=no
[ "$DEVICE_TYPE" = "cpu-simulated" ] && DEVICE_OK=yes

# Aspect 2: the numbers agree, and the reported sizes match the reported mode.
# A `constant` naming eight sizes, or a `range` naming one, means the table was
# assembled rather than observed.
#
# Scored against the tolerance the task fixed, not the one the run reported:
# comparing a run's own two numbers lets it grade itself either way. The
# reported value still has to be that number, so the row stays a check.
#
# This is a format and plausibility check, not an accuracy one. The agent
# reports its own difference, and a correct float32 CPU matmul against a
# CPU reference is near zero by construction, so what this catches is a
# placeholder, a negative, or a number that missed the fixed tolerance --
# not a wrong matmul.
#
# `max_abs_diff` must be a finite non-negative number, matched as one rather
# than as a bag of numeric characters: `-`, `+` and `.` all passed the looser
# pattern and awk read them as 0, and a negative like `-1e9` passed the upper
# bound while looking like a measurement.
NUM_RE='^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$'
TOLERANCE_SPEC=1e-3
NUMERIC_OK=no
if [ -n "$MAX_ABS_DIFF" ] && [ -n "$TOLERANCE" ]; then
  NUMERIC_OK="$(awk -v d="$MAX_ABS_DIFF" -v t="$TOLERANCE" \
      -v spec="$TOLERANCE_SPEC" -v re="$NUM_RE" '
    BEGIN {
      if (d ~ re && t ~ re && d+0 <= spec+0 && t+0 == spec+0)
        print "yes"; else print "no"
    }')"
fi
# The decision the task withholds. Shape alone cannot catch an agent that asks,
# is told 1..4, and then reports a sweep of its own choosing, which the
# instruction calls a failure however good the numbers are. Overridable so the
# task still works behind a user who chooses differently.
EXPECTED_MODE="${EXPECTED_BATCH_MODE:-range}"
EXPECTED_SIZES="${EXPECTED_BATCH_SIZES:-1,2,3,4}"
SHAPE_OK=no
case "$BATCH_MODE" in
constant) [[ "$BATCH_SIZES" =~ ^[0-9]+$ ]] && SHAPE_OK=yes ;;
range) [[ "$BATCH_SIZES" =~ ^[0-9]+(,[0-9]+)+$ ]] && SHAPE_OK=yes ;;
esac
ANSWER_OK=no
[ "$BATCH_MODE" = "$EXPECTED_MODE" ] && [ "$BATCH_SIZES" = "$EXPECTED_SIZES" ] &&
  ANSWER_OK=yes

# Aspect 3: the agent asked, and used what it was told.
#
# Keyed on the harness's own record of the ask rather than on the agent's prose
# about asking, which a run that never asked will still happily produce. The
# tool is named differently per harness, so all three spellings are accepted;
# a `needs_input` result counts too, since a specialist handing the question up
# is the designed path. Both are matched as the shape the harness writes -- a
# tool *call*, a status *value* -- because the bare names are also prose in
# several agent definitions and in any log that lists the available tools.
ASKED=no
# A call, not a mention. The bare names appear in agent definitions, in docs
# that show an example result, and in any log that lists the tool schemas, so
# `read_bridge.py logs` parses the coding agent's structured logs (and not the
# user agent's) and rejects schema definitions and `Read` tool_result blocks.
if python3 "$(dirname "$0")/read_bridge.py" logs "${AGENT_LOGS_DIR:-/logs/agent}"; then
  ASKED=yes
fi
# Over a bridge the ask is not a tool call: the agent writes the question in
# its reply and the user answers with another prompt. The evidence is the
# user's answer -- a turn after the handover, before the agent commits to a
# size, that carries a decision. Keyed on the answer rather than on the
# agent's phrasing: an agent that writes "I need you to choose the batch size"
# asked, and no question mark appears in it. Counting turns instead let three
# things pass as an ask -- one prompt counted twice across the two acpx
# renderings, a bare acknowledgement, and the user's own closing request.
if [ "$ASKED" = no ] && [ -f "$BRIDGE" ] &&
    python3 "$(dirname "$0")/read_bridge.py" asked "$BRIDGE"; then
  ASKED=yes
fi
# Reject a run where the coding agent peeked at the simulated user's logs or
# process environment instead of getting the batch size from the user.
if [ "$ASKED" = yes ] &&
    python3 "$(dirname "$0")/read_bridge.py" peeked "${AGENT_LOGS_DIR:-/logs/agent}"; then
  ASKED=no
fi

d=0; [ "$DEVICE_OK" = yes ] && d=1
n=0; [ "$NUMERIC_OK" = yes ] && [ "$SHAPE_OK" = yes ] && [ "$ANSWER_OK" = yes ] && n=1
a=0; [ "$ASKED" = yes ] && a=1
# All three. A correct sweep the user never chose is the failure this task was
# written to catch, so the ask is not a bonus aspect.
r=0; [ "$d" = 1 ] && [ "$n" = 1 ] && [ "$a" = 1 ] && r=1

printf '%s\n' "$r" >"$LOGS/reward.txt"
cat >"$LOGS/reward.json" <<JSON
{"reward": $r, "cpu_path": $d, "numerics": $n, "asked_the_user": $a}
JSON

if grep -qh '"tpu-hello-world":[1-9]' "${AGENT_LOGS[@]}" 2>/dev/null; then
  USED_PACKAGE=yes
else
  USED_PACKAGE=no
fi

echo "Device type:          ${DEVICE_TYPE:-<no table row>}"
echo "Batch mode:           ${BATCH_MODE:-<no table row>}"
echo "Batch sizes:          ${BATCH_SIZES:-<no table row>}"
echo "Max abs diff:         ${MAX_ABS_DIFF:-<no table row>}"
echo "Tolerance:            ${TOLERANCE:-<no table row>} (task fixes $TOLERANCE_SPEC)"
echo "CPU path deliberate:  $DEVICE_OK"
echo "Numerics within tol:  $NUMERIC_OK"
echo "Mode matches sizes:   $SHAPE_OK"
echo "Used the user's answer: $ANSWER_OK (expected $EXPECTED_MODE $EXPECTED_SIZES)"
echo "Asked the user:       $ASKED"
echo "Used packaged agent:  $USED_PACKAGE"
echo "Verifier final score: $(cat "$LOGS/reward.txt")"
