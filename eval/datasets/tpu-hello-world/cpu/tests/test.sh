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

# The baseline's verifier asks one question -- did a TPU do the arithmetic.
# This one cannot ask that, so it asks the two that are left: was the CPU path
# taken deliberately, and does the arithmetic agree with the reference.
#
# `native-tpu` is a FAILURE here, not a bonus. This task's whole purpose is to
# be runnable where no accelerator exists; a run reporting silicon either found
# hardware the task does not describe or invented the row.
set -euo pipefail

LOGS="${LOGS_DIR:-/logs/verifier}"
mkdir -p "$LOGS"

AGENT_LOGS=("${AGENT_LOGS_DIR:-/logs/agent}"/*)

# Lifted from the baseline verifier, and for the same reasons: unanchored
# because the agent log is JSONL and a table row never starts a physical line,
# and LAST occurrence because a session records its failed probes as well as
# its result.
table_row() {
  { grep -ohE "\|[[:space:]]*$1[[:space:]]*\|[[:space:]]*[^|[:space:]][^|]*\|" \
      "${AGENT_LOGS[@]}" 2>/dev/null || true; } | tail -n 1 |
    sed -E "s/.*\|[[:space:]]*$1[[:space:]]*\|[[:space:]]*//; s/\|\$//" |
    tr -d '[:space:]'
}

DEVICE_TYPE="$(table_row device_type)"
MAX_ABS_DIFF="$(table_row max_abs_diff)"
TOLERANCE="$(table_row tolerance)"

# Aspect 1: the CPU path was taken on purpose. `cpu-fallback` means MOCK_TPU
# was never set and the runtime landed on the CPU by accident -- the same
# result for a different reason, which is exactly the confusion this row
# exists to prevent.
DEVICE_OK=no
[ "$DEVICE_TYPE" = "cpu-simulated" ] && DEVICE_OK=yes

# Aspect 2: the numbers agree. Compared numerically, not as strings: "0.000000"
# and "0.0" are the same answer, and awk is the only calculator guaranteed to
# be in this image.
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

d=0; [ "$DEVICE_OK" = yes ] && d=1
n=0; [ "$NUMERIC_OK" = yes ] && n=1
# Both or nothing: a correct number on the wrong device says the task was not
# run, and the right device with a bad number says PyTorch is broken.
r=0; [ "$d" = 1 ] && [ "$n" = 1 ] && r=1

printf '%s\n' "$r" >"$LOGS/reward.txt"
cat >"$LOGS/reward.json" <<JSON
{"reward": $r, "cpu_path": $d, "numerics": $n}
JSON

# Reported, not scored -- the same A/B signal the baseline records, keyed on
# the CLI's own subagent accounting rather than on a script filename, which a
# control arm will happily reuse.
if grep -qh '"tpu-hello-world":[1-9]' "${AGENT_LOGS[@]}" 2>/dev/null; then
  USED_PACKAGE=yes
else
  USED_PACKAGE=no
fi

echo "Device type:          ${DEVICE_TYPE:-<no table row>}"
echo "Max abs diff:         ${MAX_ABS_DIFF:-<no table row>}"
echo "Tolerance:            ${TOLERANCE:-<no table row>}"
echo "CPU path deliberate:  $DEVICE_OK"
echo "Numerics within tol:  $NUMERIC_OK"
echo "Used packaged agent:  $USED_PACKAGE"
echo "Verifier final score: $(cat "$LOGS/reward.txt")"
