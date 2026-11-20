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
"""Trajectory criteria for preflight-ambiguous-name.

"llama 3 8b" names several official repositories (Llama 3 and 3.1, base and
instruct). Preflight resolves the name, finds it ambiguous, and hands the
question back; no check runs on a repository it guessed.
"""

import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from preflight_common import (commands, is_full_check, is_hub_only, is_resolve,
                              on_accelerator)
from rewardkit import criterion


@criterion(description="Preflight resolved the loose name with --resolve")
def resolved_name(workspace: Path) -> bool:
    return any(is_resolve(c) and "llama" in c.lower() for c in commands())


@criterion(description="No check ran on a repository chosen without the user")
def no_check_on_a_guess(workspace: Path) -> bool:
    return not any(
        is_hub_only(c) or is_full_check(c) or on_accelerator(c)
        for c in commands())
