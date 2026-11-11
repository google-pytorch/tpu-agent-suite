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

set -eo pipefail

ROLE="${ROLE:-decode}"
PREFILL_TP_SIZE="${PREFILL_TP_SIZE:-8}"
PREFILL_PCP_SIZE="${PREFILL_PCP_SIZE:-2}"
PREFILL_DCP_SIZE="${PREFILL_DCP_SIZE:-16}"
DECODE_TP_SIZE="${DECODE_TP_SIZE:-8}"
DECODE_PCP_SIZE="${DECODE_PCP_SIZE:-2}"
DECODE_DCP_SIZE="${DECODE_DCP_SIZE:-16}"

export TPU_FUSED_QK_NORM_ROPE=1
export TPU_LATENT_PROJ_INTRA_CHIP_TP=1
export TPU_SMALL_ALL_REDUCE_MAX_ROWS=128
export TPU_SMALL_ALL_REDUCE_FORM=oneshot

if [[ "${ROLE}" == "prefill" ]]; then
  exec vllm serve zai-org/GLM-5-FP8 \
    --port 8100 \
    --tensor-parallel-size "${PREFILL_TP_SIZE}" \
    --prefill-context-parallel-size "${PREFILL_PCP_SIZE}" \
    --decode-context-parallel-size "${PREFILL_DCP_SIZE}" \
    --kv-cache-dtype fp8 \
    --kv-cache-memory-bytes 36000000000 \
    --max-model-len 202752 \
    --max-num-seqs 32 \
    --enable-prefix-caching \
    --async-scheduling \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --kv-transfer-config '{"kv_connector":"TpuRaidenConnector","kv_role":"kv_producer"}'
else
  exec vllm serve zai-org/GLM-5-FP8 \
    --port 8200 \
    --tensor-parallel-size "${DECODE_TP_SIZE}" \
    --prefill-context-parallel-size "${DECODE_PCP_SIZE}" \
    --decode-context-parallel-size "${DECODE_DCP_SIZE}" \
    --kv-cache-dtype fp8 \
    --kv-cache-memory-bytes 36000000000 \
    --max-model-len 202752 \
    --max-num-seqs 32 \
    --enable-prefix-caching \
    --async-scheduling \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    --kv-transfer-config '{"kv_connector":"TpuRaidenConnector","kv_role":"kv_consumer"}'
fi
