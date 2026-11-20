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
"""Trajectory criteria for preflight-gptq-host-reject.

A GPTQ checkpoint fails the quantization rule in the Hub-side checks, which
run on the host. Preflight stops there: no image probe, no full check, no
chip.
"""

import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from preflight_common import (commands, first_index, is_full_check,
                              is_hub_only, is_resolve, on_accelerator)
from rewardkit import criterion


@criterion(description="The model id was resolved before any check ran on it")
def resolved_first(workspace: Path) -> bool:
    cmds = commands()
    resolve = first_index(cmds, is_resolve)
    check = first_index(cmds, lambda c: is_hub_only(c) or is_full_check(c))
    return resolve is not None and (check is None or resolve < check)


@criterion(description="The Hub-side checks ran on the host with --hub_only")
def hub_checks_on_host(workspace: Path) -> bool:
    return any(is_hub_only(c) and not on_accelerator(c) for c in commands())


@criterion(description="Rejected on the host: nothing ran on the accelerator")
def rejected_before_chip(workspace: Path) -> bool:
    cmds = commands()
    return (any(is_hub_only(c) for c in cmds)
            and not any(on_accelerator(c) for c in cmds))
