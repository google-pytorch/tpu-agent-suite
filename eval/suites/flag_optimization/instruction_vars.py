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
"""The `prompt_vars` of each case, derived from its task in tasks.json.

assets/instruction.md is one template for every task; flag_optimization.yaml
gives each case the values for its placeholders. This module computes those
values from the task spec, so they cannot drift from what `flagopt` enforces:
eval/tests/test_flag_optimization_suite.py checks the YAML against it.

    python3 eval/suites/flag_optimization/instruction_vars.py [task_id]

prints the values to paste into the YAML after changing tasks.json.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

SUITE_DIR = Path(__file__).resolve().parent
CATALOG = SUITE_DIR / "assets" / "flagopt" / "tasks.json"

GROUP_LABEL = {
    "sharding": "Sharding",
    "remat": "Rematerialization",
    "framework": "MaxText framework flags",
    "kernel": "Kernel hyperparameters",
}
SHARDING_ABBR = {
    "ici_data_parallelism": "DP",
    "ici_fsdp_parallelism": "FSDP",
    "ici_fsdp_transpose_parallelism": "FSDP-transpose",
    "ici_tensor_parallelism": "TP",
    "ici_tensor_sequence_parallelism": "TP-sequence",
    "ici_context_parallelism": "CP",
    "ici_expert_parallelism": "EP",
}
SCOPE = {
    "remat": ("activation rematerialization (remat policy, or a custom "
              "per-tensor remat / device / host-offload plan)"),
    "xla":
    "XLA / libtpu compiler flags (`LIBTPU_INIT_ARGS`)",
    "framework":
    ("MaxText framework flags (embedding / vocab tiling, host offload of "
     "optimizer state or parameters, gradient accumulation with the global "
     "batch held constant, and MoE dispatch options where they apply)"),
    "kernel":
    ("kernel hyperparameters (splash-attention block sizes, layouts "
     "and backward kernel; grouped-matmul tile sizes for MoE models)"),
}
DTYPE = {"bfloat16": "bf16", "float32": "fp32", "float16": "fp16"}
XLA_TUNABLE = (
    "Each entry of `xla_flags` is one `--xla_...=value` (or `--libtpu_...`/"
    "`--megascale_...`) flag, passed to the TPU runtime as `LIBTPU_INIT_ARGS`. "
    "Debug / dump / SDC-checker / CPU / GPU flags are rejected. Note: on "
    "TPU7x, `--xla_tpu_enable_async_collective_fusion=true` (continuation "
    "fusion) is not supported and fails the compile.")
XLA_FIXED = "XLA flags are not tunable here: leave `xla_flags` empty."


def load_flagopt():
    """The flagopt module of this suite (it lives in the task image).

    Loaded without writing bytecode: assets/flagopt/ is copied into the image
    as is, and a stray __pycache__ there would change its content hash.
    """
    spec = importlib.util.spec_from_file_location(
        "flagopt", SUITE_DIR / "assets" / "flagopt" / "flagopt.py")
    module = importlib.util.module_from_spec(spec)
    previous = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def task_specs() -> dict[str, dict[str, Any]]:
    """Every task of the catalog, resolved by flagopt.expand_task."""
    flagopt = load_flagopt()
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    return {t: flagopt.expand_task(catalog, t) for t in catalog["tasks"]}


def prompt_vars(spec: dict[str, Any]) -> dict[str, str]:
    """Placeholder values of assets/instruction.md for one task."""
    hw, model, batch, fixed = (spec["hardware"], spec["model"], spec["batch"],
                               spec["fixed"])
    tun = spec["tunables"]
    by_group: dict[str, list[str]] = {}
    for name, knob in tun.items():
        by_group.setdefault(knob["group"], []).append(name)

    sharding = by_group.get("sharding", [])
    if sharding == [
            "ici_data_parallelism", "ici_fsdp_parallelism",
            "ici_tensor_parallelism"
    ]:
        scope = [("the device mesh: data (DP), fully-sharded data (FSDP) and "
                  "tensor (TP) parallelism degrees")]
    else:
        scope = [
            "sharding (" + ", ".join(SHARDING_ABBR[k] for k in sharding) + ")"
        ]
    xla = spec["xla_flags"]["tunable"]
    for group in ("remat", "xla", "framework", "kernel"):
        if (group == "xla" and xla) or group in by_group:
            scope.append(SCOPE[group])

    knobs = [
        f"- {GROUP_LABEL[g]}: " + ", ".join(f"`{k}`" for k in by_group[g])
        for g in ("sharding", "remat", "framework", "kernel") if g in by_group
    ]
    rules = []
    if "per_device_batch_size" in tun or "gradient_accumulation_steps" in tun:
        product = batch["per_device_batch_size"] * batch[
            "gradient_accumulation_steps"]
        rules.append("- When you change `per_device_batch_size` or "
                     "`gradient_accumulation_steps`, their product must stay "
                     f"{product:g} so the global batch is unchanged.")
    if "remat" in by_group:
        rules.append("- The per-tensor remat knobs only take effect with "
                     "`remat_policy=custom`.")
    rules_text = ("\n".join(rules) + "\n\n"
                  if rules else "") + (XLA_TUNABLE if xla else XLA_FIXED)
    hours = spec["time_budget_sec"] / 3600
    return {
        "model":
        model["display"].split(" (")[0].strip(),
        "slice":
        f"{hw['accelerator_display']} `{hw['topology']}`",
        "scope":
        "\n".join(f"- {s}" for s in scope),
        "budget":
        f"{hours:g} hour" + ("" if hours == 1 else "s"),
        "note":
        f"\n\nNote: {spec['notes']}" if spec.get("notes") else "",
        "maxtext_model":
        model["maxtext_model_name"],
        "hf_repo":
        model["hf_repo"],
        "batch":
        f"{batch['global_batch_size']} sequences x {batch['seq_len']} tokens",
        "precision": (f"{DTYPE[fixed['dtype']]} activations, "
                      f"{DTYPE[fixed['weight_dtype']]} weights, "
                      f"{DTYPE[fixed['grad_dtype']]} gradients"),
        "topology":
        hw["topology"],
        "knobs":
        "\n".join(knobs),
        "rules":
        rules_text,
    }


def main(argv: list[str]) -> int:
    specs = task_specs()
    for task_id in argv or list(specs):
        print(f"# {task_id}")
        print(json.dumps(prompt_vars(specs[task_id]), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
