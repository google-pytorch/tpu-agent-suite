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

# Harbor reads one number from /logs/verifier/reward.txt. For a tpu-hello-world
# that is the whole verifier -- no rewardkit, no criteria files, no judge.
#
# The number is the hardware result and nothing else: did a TPU actually do the
# arithmetic. Whether the agent got there by invoking `tpu-hello-world` or by writing
# its own manifest is reported alongside it, not folded into it -- both arms are
# given the same cluster and the same image, so both can legitimately pass, and
# a score that conflated the two would say the package failed whenever the
# control arm succeeded.
set -euo pipefail

LOGS="${LOGS_DIR:-/logs/verifier}"
mkdir -p "$LOGS"

# The pod's output reaches the host through the agent's captured session: the
# agent runs `kubectl logs`, so what the TPU printed is quoted verbatim in
# there. It has to be read this way rather than from the cluster, because the
# Job is deleted once its logs have been read -- a Job left holding a TPU node
# keeps the next run Pending.
AGENT_LOGS=(/logs/agent/*)

# The LAST occurrence, not any occurrence. An agent's session records its
# exploration as well as its result: probes that failed, approaches abandoned,
# and the instructions it read along the way, which quote every device label as
# prose. Scoring "does this string appear anywhere" makes a run that succeeded
# after a false start indistinguishable from one that never succeeded.
# Read the row out of the result table the prompt requires, not a string only
# the packaged script prints. The control arm can produce this table too, which
# is the point: the reward measures the hardware result, not which agent ran.
table_row() {
  # Unanchored: the agent log is JSONL, so the table lives inside JSON strings
  # with escaped newlines and a row never starts a physical line. The `| name |`
  # structure is what makes the match specific -- that is what stops a bare
  # generation string in prose, or the cluster's own name, from satisfying it.
  { grep -ohE "\|[[:space:]]*$1[[:space:]]*\|[[:space:]]*[^|[:space:]]+" "${AGENT_LOGS[@]}" 2>/dev/null || true; } |
    tail -n 1 | sed -E "s/.*\|[[:space:]]*$1[[:space:]]*\|[[:space:]]*//" | tr -d '[:space:]'
}

DEVICE_TYPE="$(table_row device_type)"
ACCELERATOR="$(table_row accelerator)"
SPEC_SOURCE="$(table_row spec_source)"
XPROF_CLAIMED="$(table_row xprof_captured)"

# The generation the run actually landed on, reduced to the token a vendor spec
# page is named after: `v6e-8` -> `v6e`, `tpu7x-8` -> `tpu7x`, and the node
# label form `tpu-v6e-slice` -> `v6e` in case that is what got reported.
GENERATION="$(printf '%s' "$ACCELERATOR" | tr '[:upper:]' '[:lower:]' |
  sed -E 's/^tpu-//; s/-slice$//; s/-[0-9]+$//')"

# Is the cited page about the chip that ran? A URL is easy to produce and easy
# to produce for the wrong generation -- these spec pages are near-identical in
# layout and differ in exactly the numbers, so a v5e page cited for a v6e run
# looks entirely plausible and every figure under it is wrong. Matching the
# generation token against the URL is a weak check by design: it cannot tell
# whether the page was read, only that it is not about a different chip.
SPEC_OK=no
SPEC_WHY=""
case "$SPEC_SOURCE" in
"")
  SPEC_WHY="no spec_source row in the reported table"
  ;;
unreachable | not-stated | not_stated | none | n/a)
  SPEC_WHY="spec_source reported as '$SPEC_SOURCE' -- honest, but no citation was produced"
  ;;
http://* | https://*)
  if [ -z "$GENERATION" ]; then
    SPEC_WHY="cannot check the URL: no accelerator row to compare against"
  elif printf '%s' "$SPEC_SOURCE" | tr '[:upper:]' '[:lower:]' | grep -qF "$GENERATION"; then
    SPEC_OK=yes
  else
    SPEC_WHY="spec_source does not mention '$GENERATION' -- cited page may be for another generation"
  fi
  ;;
*)
  SPEC_WHY="spec_source is not a URL"
  ;;
esac

# One question: did a real TPU do the arithmetic. `native-tpu`
# is the only label where it landed on silicon -- "cpu-fallback" and
# "cpu-simulated" (MOCK_TPU=1) did not.
#
# The chip generation is reported but not scored. Gating on it would pin the
# task to one cluster: this one is tpu7x, but the same check is meant to run
# against v6e and whatever comes next.
#
# The value is read out of the reported table, so it cannot be satisfied by
# prose or by the cluster's own name -- which matters, since this cluster is
# called `torchtpu-distributed-v7x-8`.
# Did a profile actually land? Checked on disk, not from the reported row: a
# capture that returns cleanly and writes nothing reads as success everywhere
# except here, and `xprof_captured: yes` is the agent's claim about its own
# work. An .xplane.pb of zero bytes is the exact shape of that failure, so
# emptiness disqualifies a file rather than merely being noted.
ARTIFACTS="${ARTIFACTS_DIR:-/logs/artifacts}"
XPLANES=0
if [ -d "$ARTIFACTS" ]; then
  XPLANES="$(find "$ARTIFACTS" -name '*.xplane.pb' -size +0 2>/dev/null | wc -l | tr -d '[:space:]')"
  OMITTED_XPLANES="$(python3 - "$ARTIFACTS" <<'PYEOF'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
n = 0
for p in root.rglob("*.xplane.pb.omitted.json"):
    try:
        if int(json.loads(p.read_text(encoding="utf-8")).get("size_bytes", 0)) > 0:
            n += 1
    except Exception:
        pass
print(n)
PYEOF
)"
  XPLANES=$((XPLANES + OMITTED_XPLANES))
fi

XPROF_OK=no
XPROF_WHY=""
if [ "$XPLANES" -gt 0 ]; then
  XPROF_OK=yes
elif [ "$XPROF_CLAIMED" = "yes" ]; then
  XPROF_WHY="table says yes but no non-empty .xplane.pb under $ARTIFACTS"
else
  XPROF_WHY="no non-empty .xplane.pb under $ARTIFACTS"
fi

# Scored per aspect, not all-or-nothing. Harbor reads reward.json in
# preference to reward.txt and takes a {name: number} map, so each thing the
# task asks for gets its own number and the overall is their weighted sum.
# A single pass/fail cannot say whether a run missed the hardware, the citation
# or the trace, and those call for three different fixes.
#
# Equal thirds: this task is the cited task plus profiling, and all three are
# things it asks for. Change the weights, not the gates, if that stops being
# true.
HARDWARE_OK=no
case "$DEVICE_TYPE" in
"native-tpu") HARDWARE_OK=yes ;;
esac

h=0; [ "$HARDWARE_OK" = yes ] && h=1
s=0; [ "$SPEC_OK" = yes ] && s=1
x=0; [ "$XPROF_OK" = yes ] && x=1
REWARD="$(awk -v h="$h" -v s="$s" -v x="$x" \
  'BEGIN { printf "%.4f", (h + s + x) / 3 }')"

printf '%s\n' "$REWARD" >"$LOGS/reward.txt"
cat >"$LOGS/reward.json" <<JSON
{"reward": $REWARD, "hardware": $h, "spec_source": $s, "xprof": $x}
JSON

# The second signal, reported and not scored. An installed subagent is an offer
# rather than an instruction, and a capable model handed a cluster and a clear
# question will often just do the work itself -- so this is the measurement the
# A/B is actually for.
#
# Keyed on the CLI's own subagent accounting, which appears in the terminal
# result event as `"by_type":{"tpu-hello-world":N}`. The obvious grep for the script
# filename reported `yes` for a run with no package installed at all: the
# control arm wrote its own check and, taking the name from the prompt, called
# it hello_tpu.py.
if grep -qh '"tpu-hello-world":[1-9]' "${AGENT_LOGS[@]}" 2>/dev/null; then
  USED_PACKAGE=yes
else
  USED_PACKAGE=no
fi

echo "Device type:         ${DEVICE_TYPE:-<no table row>}"
echo "Accelerator:         ${ACCELERATOR:-<no table row>}"
echo "Hardware ran on TPU: $HARDWARE_OK"
echo "Spec source:         ${SPEC_SOURCE:-<no table row>}"
echo "Spec source valid:   $SPEC_OK${SPEC_WHY:+  ($SPEC_WHY)}"
echo "xprof claimed:       ${XPROF_CLAIMED:-<no table row>}"
echo "xplane files found:  $XPLANES"
echo "xprof captured:      $XPROF_OK${XPROF_WHY:+  ($XPROF_WHY)}"
echo "Used packaged agent: $USED_PACKAGE"
echo "Verifier final score: $(cat "$LOGS/reward.txt")  ($LOGS/reward.json carries the per-aspect breakdown)"
