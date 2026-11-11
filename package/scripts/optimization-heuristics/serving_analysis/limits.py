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
"""Shared vLLM-TorchTPU kernel row limits and MLA/DCP guard thresholds.

Consumed by both `serving_analysis` detectors (`D18`, `D20`, `D31`, `D52`) and
the `hillclimb` preflight fallback when a live target checkout is not passed.
"""

# vLLM-TorchTPU `moe_tp_decode` kernel ROW_BUCKETS on upstream main (16, 32, 64) and
# its hard upper bound. Bump both when the kernel admits more rows upstream.
MOE_TP_DECODE_ROW_BUCKETS: tuple[int, ...] = (16, 32, 64)
MOE_TP_DECODE_MAX_ROWS: int = MOE_TP_DECODE_ROW_BUCKETS[-1]

# Asymmetric P/D decode-context-parallel guardrails (`D18`, `preflight.py`).
MIN_ASYMMETRIC_PREFILL_DCP: int = 16
SMALL_BATCH_DECODE_TARGET_DCP: int = 8
SMALL_DECODE_MAX_SEQS: int = 16
SMALL_DECODE_MAX_BUCKET: int = 64
# vLLM's default `dcp_kv_cache_interleave_size` (vllm/config/parallel.py): what a
# deployment runs with when it does not set the interleave.
DEFAULT_DCP_INTERLEAVE: int = 1

# MLA KV-cache sizing defaults when HF config or CLI flags omit them.
DEFAULT_MLA_KV_LORA_RANK: int = 512
DEFAULT_MLA_ROPE_DIM: int = 64
DEFAULT_MLA_LATENT_DIM: int = DEFAULT_MLA_KV_LORA_RANK + DEFAULT_MLA_ROPE_DIM
DEFAULT_KV_BUDGET_BYTES: int = 36_000_000_000
