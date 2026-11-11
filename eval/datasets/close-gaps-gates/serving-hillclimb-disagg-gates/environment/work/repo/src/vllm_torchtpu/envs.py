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
"""Environment variable registry for vllm-torchtpu."""

import os

environment_variables = {
    # Controls striped KV cache page layout; set to 0 for asymmetric P/D DCP transfers.
    "TPU_KV_STRIPED_LAYOUT":
    lambda: os.getenv("TPU_KV_STRIPED_LAYOUT", "1") == "1",
    # Prefill drafter tensor-parallel override.
    "DRAFTER_TP":
    lambda: int(os.getenv("DRAFTER_TP", "8")),
    # Absorbs MLA Q head projection into decode attention; allocates 68 MiB VMEM at DCP>=16.
    "TPU_MLA_ABSORB_Q_FROM_HEAD":
    lambda: os.getenv("TPU_MLA_ABSORB_Q_FROM_HEAD", "0") == "1",
}
