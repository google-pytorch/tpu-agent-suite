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
# shellcheck shell=bash
# Sourced, not run: defines submit_bench, the client submission the onboarding
# used. $C is the compute command, $B this directory, $P the installed package
# and $SNAP the model snapshot path.

submit_bench () {
    local N=$1 ISL=$2 OSL=$3 CC=$4 DS=$5 WU=$6 SEED=$7
    local CELL=isl${ISL}_osl${OSL}_c${CC}
    # $C is a command and its arguments, so it is word-split on purpose.
    # shellcheck disable=SC2086
    $C submit --recipe "$B/bench_client.sh" \
        --input "$P/scripts/common/serving_benchmark.py:serving_benchmark.py" \
        --input "$B/$DS:$DS" \
        --output serving_bench/summary.json \
        --output serving_bench/summary.md \
        --output "serving_bench/raw/${CELL}_r1.json" \
        --output "serving_bench/raw/${CELL}_r1.log" \
        --name "$N" -- \
        --base-url http://localhost:8000 --engine vllm \
        --model google/gemma-4-31B-it --tokenizer "$SNAP" \
        --output-dir serving_bench --isl-osl "${ISL}:${OSL}" \
        --concurrency "$CC" --replicates 1 \
        --num-prompts "$(wc -l < "$B/$DS")" --num-warmups "$WU" --seed "$SEED" \
        --run-timeout 14400 \
        --extra-bench-args "--dataset-name custom --dataset-path $DS --custom-output-len $OSL --skip-chat-template --no-oversample --save-detailed"
}
