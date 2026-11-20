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
"""Trajectory criteria for preflight-full-sequence.

openai/gpt-oss-20b on a host without transformers: every step runs, the cheap
ones first, and the full check goes to the accelerator because the host
cannot run it.
"""

import sys
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from preflight_common import (artifact, commands, first_index,
                              installed_transformers, is_full_check,
                              is_hub_only, is_probe, is_resolve,
                              on_accelerator, read_json)
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


@criterion(
    description="The Hub-side checks ran before anything queued for a chip")
def hub_checks_before_chip(workspace: Path) -> bool:
    cmds = commands()
    hub = first_index(cmds, is_hub_only)
    chip = first_index(cmds, on_accelerator)
    return hub is not None and chip is not None and hub < chip


@criterion(
    description="The task image was probed with image_probe.py on the "
    "cluster, and the probe found torch_tpu, a non-failing device check "
    "and a vLLM stack verdict")
def image_probed(workspace: Path) -> bool:
    if not any(is_probe(c) for c in commands()):
        return False
    # Only tpu-model-onboarding reads the probe's report, so where under the
    # artifacts it landed does not matter; what the probe found does.
    reports = [read_json(p) for p in artifact("image_probe.json", nested=True)]

    def found(report: dict) -> bool:
        stack = (report.get("dependencies") or {}).get("vllm_stack") or {}
        return (report.get("torch_tpu_present") is True and
                (report.get("device") or {}).get("status") in ("ok", "skipped")
                and "verdict" in stack)

    return any(found(r) for r in reports)


@criterion(
    description="Without transformers on the host, the full check ran "
    "on the accelerator, and nothing installed transformers to avoid it")
def full_check_on_accelerator(workspace: Path) -> bool:
    cmds = commands()
    return (any(is_full_check(c) and on_accelerator(c) for c in cmds)
            and not any(installed_transformers(c) for c in cmds))


@criterion(description="The model's own CUDA kernels report was written")
def cuda_report_written(workspace: Path) -> bool:
    reports = [
        read_json(p)
        for p in artifact("cuda_kernels_report.openai--gpt-oss-20b.json")
    ]
    return any(r.get("model_id") == "openai/gpt-oss-20b" for r in reports)
