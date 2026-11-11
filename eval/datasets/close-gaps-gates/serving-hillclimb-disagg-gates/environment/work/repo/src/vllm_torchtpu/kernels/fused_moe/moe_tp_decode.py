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
"""Fused TP decode MoE kernel dispatch for TPU serving."""

ROW_BUCKETS = (16, 32, 64)


def supports_fused_tp_decode(num_tokens: int) -> bool:
    """Returns True when `num_tokens` fits a compiled fused TP decode bucket."""
    if num_tokens > 64:
        return False
    return any(bucket >= num_tokens for bucket in ROW_BUCKETS)
