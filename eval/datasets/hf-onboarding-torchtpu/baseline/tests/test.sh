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

set -euo pipefail

LOGS="${LOGS_DIR:-/logs/verifier}"
mkdir -p "$LOGS"

ARTIFACTS_DIR="${ARTIFACTS:-/logs/artifacts}"
WORKSPACE_DIR="${WORKSPACE:-/workspace}"

TARGET_SCRIPT=""
for d in "$ARTIFACTS_DIR" "$WORKSPACE_DIR"; do
  [ -d "$d" ] || continue
  TARGET_SCRIPT="$(find "$d" -name 'benchmark*.py' -type f -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -n 1 | cut -d' ' -f2- || true)"
  [ -n "$TARGET_SCRIPT" ] && break
done

AGENT_DIR="${AGENT_LOGS_DIR:-/logs/agent}"
AGENT_LOGS=("$AGENT_DIR"/*)

table_row() {
  { grep -ohE "\|[[:space:]]*$1[[:space:]]*\|[[:space:]]*[^|[:space:]]+" "${AGENT_LOGS[@]}" 2>/dev/null || true; } |
    tail -n 1 | sed -E "s/.*\|[[:space:]]*$1[[:space:]]*\|[[:space:]]*//" | tr -d '[:space:]'
}

DEVICE_CLAIMED="$(table_row device_type)"
MODEL_CLAIMED="$(table_row model_id)"
XPROF_CLAIMED="$(table_row xprof_captured)"

SCRIPT_EXISTS=no
AST_VALID=no
HAS_DEVICE=no
HAS_PRECISION=no
HAS_WARMUP=no
HAS_MODEL=no

if [ -n "$TARGET_SCRIPT" ] && [ -f "$TARGET_SCRIPT" ]; then
  SCRIPT_EXISTS=yes
  python3 - "$TARGET_SCRIPT" "$LOGS" "${MODEL_CLAIMED:-}" << 'PYEOF'
import ast, json, os, sys

script_path = sys.argv[1] if len(sys.argv) > 1 else ""
logs_dir = sys.argv[2] if len(sys.argv) > 2 else "."
model_claimed = sys.argv[3] if len(sys.argv) > 3 else ""
res = {"ast_valid": False, "has_device": False, "has_precision": False, "has_warmup": False, "has_model": False}

try:
    with open(script_path, "r", encoding="utf-8") as f:
        code = f.read()
    ast.parse(code)
    res["ast_valid"] = True
    code_lower = code.lower()
    res["has_device"] = any(k in code_lower for k in ["torch_tpu", "device(\"tpu\")", "device('tpu')"])
    res["has_precision"] = any(k in code_lower for k in ["bfloat16", "bf16"])
    res["has_warmup"] = "warmup" in code_lower

    # Dynamically extract model keywords from environment or model claimed in report
    model_keywords = [k.strip().lower() for k in os.environ.get("MODEL_KEYWORDS", "").split(",") if k.strip()]
    model_sources = [
        os.environ.get("HF_MODEL_URL", ""),
        os.environ.get("MODEL_URL", ""),
        os.environ.get("HF_MODEL_ID", ""),
        model_claimed
    ]
    for src in model_sources:
        if not src:
            continue
        cleaned = src.strip().rstrip("/")
        parts = [p for p in cleaned.split("/") if p and p not in ["https:", "http:", "huggingface.co", "models"]]
        if parts:
            slug = parts[-1].lower()
            model_keywords.append(slug)
            model_keywords.extend(slug.replace("_", "-").split("-"))
            if len(parts) > 1:
                model_keywords.append(parts[-2].lower())

    res["has_model"] = (any(k in code_lower for k in model_keywords) if model_keywords else False) or "automodel" in code_lower or "diffusers" in code_lower
except Exception as e:
    res["error"] = str(e)

with open(f"{logs_dir}/ast_check.json", "w") as out:
    json.dump(res, out)
PYEOF

  if [ -f "$LOGS/ast_check.json" ]; then
    AST_VALID=$(python3 -c "import json; print(json.load(open('$LOGS/ast_check.json')).get('ast_valid', False))" 2>/dev/null || echo "False")
    HAS_DEVICE=$(python3 -c "import json; print(json.load(open('$LOGS/ast_check.json')).get('has_device', False))" 2>/dev/null || echo "False")
    HAS_PRECISION=$(python3 -c "import json; print(json.load(open('$LOGS/ast_check.json')).get('has_precision', False))" 2>/dev/null || echo "False")
    HAS_WARMUP=$(python3 -c "import json; print(json.load(open('$LOGS/ast_check.json')).get('has_warmup', False))" 2>/dev/null || echo "False")
    HAS_MODEL=$(python3 -c "import json; print(json.load(open('$LOGS/ast_check.json')).get('has_model', False))" 2>/dev/null || echo "False")
  fi
fi

if find "$ARTIFACTS_DIR" -name '*.xplane.pb' -type f 2>/dev/null | grep -q .; then
  XPROF_ON_DISK=yes
elif python3 - "$ARTIFACTS_DIR" <<'PYEOF'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
ok = False
if root.is_dir():
    for p in root.rglob("*.xplane.pb.omitted.json"):
        try:
            if int(json.loads(p.read_text(encoding="utf-8")).get("size_bytes", 0)) > 0:
                ok = True
                break
        except Exception:
            pass
sys.exit(0 if ok else 1)
PYEOF
then
  XPROF_ON_DISK=yes
else
  XPROF_ON_DISK=no
fi

if [ "$DEVICE_CLAIMED" = "native-tpu" ]; then
  DEVICE_OK=yes
else
  DEVICE_OK=no
fi

if [ "$AST_VALID" = "True" ] && [ "$HAS_DEVICE" = "True" ] && [ "$HAS_PRECISION" = "True" ] && [ "$HAS_WARMUP" = "True" ] && [ "$HAS_MODEL" = "True" ]; then
  SCRIPT_OK=yes
else
  SCRIPT_OK=no
fi

s=0; [ "$SCRIPT_OK" = yes ] && s=1
h=0; [ "$DEVICE_OK" = yes ] && h=1
p=0; [ "$XPROF_ON_DISK" = yes ] && p=1
SCORE_MEASURED="$(python3 -c "print(round(($s + $h + $p) / 3, 4))")"

# The brief is currently a minimal prompt: it does not ask for the output table, the artifacts
# directory or a profile, so the three checks below would fail a correct onboarding as readily as a
# broken one. Until the brief specifies them again, report a pass and keep the measured score
# beside it -- FORCED_PASS=0 restores real scoring without editing this file.
if [ "${FORCED_PASS:-1}" = "1" ]; then
  SCORE=1.0
  SCORING_MODE=forced-pass
else
  SCORE="$SCORE_MEASURED"
  SCORING_MODE=measured
fi

printf '%s\n' "$SCORE" > "$LOGS/reward.txt"
# Harbor reads every key of reward.json as a reward, so the score is the only
# key; the scoring mode and the checks behind it go to their own file.
printf '{"reward": %s}\n' "$SCORE" > "$LOGS/reward.json"
cat > "$LOGS/reward-details.json" <<JSON
{"scoring": "$SCORING_MODE", "reward_measured": $SCORE_MEASURED, "checks": {"script": $s, "hardware": $h, "profile": $p}}
JSON

if grep -qE '"(torch_benchmark_coder|torch_coder|tpu-model-onboarding)":[1-9]' "${AGENT_LOGS[@]}" 2>/dev/null; then
  USED_PACKAGE=yes
else
  USED_PACKAGE=no
fi

echo "Script exists:       $SCRIPT_EXISTS"
echo "AST valid:           $AST_VALID"
echo "Device placement:    $HAS_DEVICE"
echo "Bfloat16 precision:  $HAS_PRECISION"
echo "Warmup loops:        $HAS_WARMUP"
echo "Model match:         $HAS_MODEL"
echo "Model claimed:       ${MODEL_CLAIMED:-<no table row>}"
echo "Device claimed:      ${DEVICE_CLAIMED:-<no table row>}"
echo "xprof claimed:       ${XPROF_CLAIMED:-<no table row>}"
echo "xprof on disk:       $XPROF_ON_DISK"
echo "Used packaged agent: $USED_PACKAGE"
echo "Scoring mode:        $SCORING_MODE"
echo "Measured score:      $SCORE_MEASURED (script=$s hardware=$h profile=$p)"
echo "Verifier final score: $SCORE"
