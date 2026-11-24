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
"""Input validators for Campaign, Protocol, Hypothesis, and `__flags.json` configs."""

from __future__ import annotations

import functools
import json
import math
import os
import pathlib
import re
import shlex
from typing import Any, Mapping

from framework import jsonio, utils
from framework.hc import constants

Path = pathlib.Path

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESERVED_CONTEXT_FLAG_KEYS = frozenset({
    "accuracy_metric",
    "base_client_args",
    "base_eval_args",
    "base_server_args",
    "client_args",
    "eval_args",
    "eval_limit",
    "eval_tasks",
    "model",
    "target_model",
    "weights_path",
})
_RESERVED_EVAL_FLAGS = (
    "--tasks",
    "-t",
    "--limit",
    "-L",
    "--model",
    "--model_args",
    "--output_path",
    "-o",
)
_INFERENCE_CELL_KEYS = ("prefill", "decode", "concurrency")
_TRAINING_CELL_KEYS = ("sequence_length", "global_batch_size",
                       "per_device_batch_size")


@functools.cache
def _load_hc_schema(schema_name: str) -> dict[str, Any]:
    return jsonio.load_json(utils.runtime_schema_path(schema_name))


def load_json_object(path: str | Path, label: str) -> dict[str, Any]:
    """Read and validate that `path` contains a JSON dictionary."""
    try:
        raw = jsonio.load_json(Path(path))
    except utils.TpuAgentSuiteError as exc:
        raise constants.ValidationError(f"invalid {label}: {exc}") from exc
    if not isinstance(raw, dict):
        raise constants.ValidationError(
            f"{label} must be a JSON object, got {type(raw).__name__}")
    return raw


def require_text(
    raw: Any,
    label: str,
    *,
    max_length: int = constants.MAX_TEXT_LENGTH,
) -> str:
    """Require a non-blank string within `max_length`, returning its stripped form."""
    if not isinstance(raw, str) or not raw.strip():
        raise constants.ValidationError(f"{label} must be a non-empty string")
    stripped = raw.strip()
    if len(stripped) > max_length:
        raise constants.ValidationError(
            f"{label} exceeds max length {max_length}")
    return stripped


def _strip_strings(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        return {k: _strip_strings(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_strip_strings(v) for v in value]
    return value


def _validate_structure(raw: Any, schema_name: str, label: str) -> Any:
    problems = utils.validate_instance(raw,
                                       _load_hc_schema(schema_name),
                                       path=label)
    if problems:
        raise constants.ValidationError("; ".join(problems))
    return _strip_strings(raw)


def _safe_script_relpath(raw: str, label: str) -> str:
    try:
        rel = utils.safe_relative(raw, label)
    except utils.TpuAgentSuiteError as exc:
        raise constants.ValidationError(str(exc)) from exc
    if not rel.parts:
        raise constants.ValidationError(
            f"{label} must name a file path, got {raw!r}")
    return rel.as_posix()


def _is_scalar_flag_value(val: Any) -> bool:
    if isinstance(val, float) and not math.isfinite(val):
        return False
    return val is None or isinstance(val, (str, int, float, bool))


def _is_non_null_scalar_flag_value(val: Any) -> bool:
    return val is not None and _is_scalar_flag_value(val)


def validate_flags_dict(raw: Any) -> dict[str, Any]:
    """Validate a parsed `__flags.json` object."""
    if not isinstance(raw, dict):
        raise constants.ValidationError(
            f"{constants.FLAGS_FILENAME} must contain a JSON dictionary")
    for key, val in raw.items():
        if not isinstance(key, str) or not key.strip():
            raise constants.ValidationError(
                f"{constants.FLAGS_FILENAME} keys must be non-empty strings")
        if key.strip() in _RESERVED_CONTEXT_FLAG_KEYS:
            raise constants.ValidationError(
                f"{constants.FLAGS_FILENAME} cannot override protocol/context key {key!r}"
            )
        if key == "env":
            if not isinstance(val, dict):
                raise constants.ValidationError(
                    f"{constants.FLAGS_FILENAME}.env must be a dictionary")
            for env_k, env_v in val.items():
                if not isinstance(env_k,
                                  str) or not _ENV_KEY_RE.fullmatch(env_k):
                    raise constants.ValidationError(
                        f"invalid environment variable name in {constants.FLAGS_FILENAME}.env: {env_k!r}"
                    )
                if not _is_non_null_scalar_flag_value(env_v):
                    raise constants.ValidationError(
                        f"{constants.FLAGS_FILENAME}.env[{env_k!r}] must be a finite scalar value"
                    )
        elif key in ("server_args", "runtime_args", "script_args"):
            if isinstance(val, str) or (isinstance(val, list) and all(
                    isinstance(item, str) for item in val)):
                continue
            raise constants.ValidationError(
                f"{constants.FLAGS_FILENAME}.{key} must be a string or list of strings"
            )
        elif _is_scalar_flag_value(val):
            continue
        elif isinstance(val, list) and all(
                _is_non_null_scalar_flag_value(item) for item in val):
            continue
        elif isinstance(val, dict) and all(
                isinstance(k, str) and bool(k.strip())
                and _is_scalar_flag_value(v) for k, v in val.items()):
            continue
        else:
            raise constants.ValidationError(
                f"unsupported value type for {constants.FLAGS_FILENAME}[{key!r}]"
            )
    return raw


def format_canonical_flags(flags: Mapping[str, Any]) -> str:
    """Serialize `__flags.json` deterministically."""
    validated = validate_flags_dict(dict(flags))
    return json.dumps(validated,
                      ensure_ascii=False,
                      allow_nan=False,
                      indent=2,
                      sort_keys=True) + "\n"


def validate_campaign_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate a Campaign Config JSON object."""
    out = _validate_structure(raw, "hc-campaign.json", "Campaign Config")
    out["owner"] = (out.get("owner") or (os.environ.get("USER") or "").strip()
                    or "unknown")
    return out


def validate_protocol_config(
    raw: dict[str, Any],
    *,
    is_microbenchmark: bool = False,
) -> dict[str, Any]:
    """Validate a Protocol Config JSON object (main E2E or microbenchmark)."""
    out = _validate_structure(raw, "hc-protocol.json", "Protocol Config")
    for k in ("max_cell_regression_percentage", "max_rounds",
              "max_barren_rounds", "target_improvement_percent",
              "accuracy_gate"):
        out.setdefault(k, None)

    is_inference = out["workload_kind"] == constants.WorkloadKind.INFERENCE
    for idx, cell in enumerate(out["measurement_cells"]):
        ctx = f"Protocol Config.measurement_cells[{idx}]"
        for k in (*_INFERENCE_CELL_KEYS, *_TRAINING_CELL_KEYS):
            cell.setdefault(k, None)
        cell["weight"] = float(cell["weight"])
        if cell["weight"] <= 0.0:
            raise constants.ValidationError(f"{ctx}.weight must be > 0.0")
        if is_inference:
            if any(cell[k] is None for k in _INFERENCE_CELL_KEYS):
                raise constants.ValidationError(
                    f"{ctx} requires prefill, decode, and concurrency when workload_kind is inference"
                )
            if any(cell[k] is not None for k in _TRAINING_CELL_KEYS):
                raise constants.ValidationError(
                    f"{ctx} forbids training fields when workload_kind is inference"
                )
        else:
            if cell["sequence_length"] is None or cell[
                    "global_batch_size"] is None:
                raise constants.ValidationError(
                    f"{ctx} requires sequence_length and global_batch_size when workload_kind is training"
                )
            if any(cell[k] is not None for k in _INFERENCE_CELL_KEYS):
                raise constants.ValidationError(
                    f"{ctx} forbids inference fields when workload_kind is training"
                )

    weight_sum = sum(c["weight"] for c in out["measurement_cells"])
    if abs(weight_sum - 1.0) > 1e-6:
        raise constants.ValidationError(
            f"Protocol Config.measurement_cells weights must sum to 1.0, got {weight_sum}"
        )

    out["min_win_percentage"] = float(out["min_win_percentage"])
    out["noise_floor_percentage"] = float(out["noise_floor_percentage"])
    if out["noise_floor_percentage"] > out["min_win_percentage"]:
        raise constants.ValidationError(
            "Protocol Config.noise_floor_percentage must be <= min_win_percentage"
        )
    if out["max_cell_regression_percentage"] is not None:
        out["max_cell_regression_percentage"] = float(
            out["max_cell_regression_percentage"])
    if out["target_improvement_percent"] is not None:
        out["target_improvement_percent"] = float(
            out["target_improvement_percent"])
        if out["target_improvement_percent"] <= 0.0:
            raise constants.ValidationError(
                "Protocol Config.target_improvement_percent must be > 0.0")

    gate = out["accuracy_gate"]
    if gate is None:
        if not is_microbenchmark:
            raise constants.ValidationError(
                "Protocol Config.accuracy_gate is required on the main E2E protocol"
            )
    else:
        gate.setdefault("relative_to", None)
        if not gate["is_relative"] and gate["relative_to"] is not None:
            raise constants.ValidationError(
                "Protocol Config.accuracy_gate.relative_to must be null when is_relative is false"
            )
        gate["threshold"] = float(gate["threshold"])
        if gate["relative_to"] is not None:
            gate["relative_to"] = float(gate["relative_to"])

    spec = out["execution_spec"]
    for k in ("microbenchmark_target", "base_server_args", "base_client_args",
              "base_eval_args"):
        spec.setdefault(k, None)
    target = spec["microbenchmark_target"]
    if is_microbenchmark:
        if target is None:
            raise constants.ValidationError(
                "Protocol Config.execution_spec.microbenchmark_target is required on microbenchmark protocols"
            )
        target = _safe_script_relpath(
            target, "Protocol Config.execution_spec.microbenchmark_target")
        if not target.endswith((".json", ".py")):
            raise constants.ValidationError(
                "Protocol Config.execution_spec.microbenchmark_target must be a .json steps plan or .py script"
            )
        spec["microbenchmark_target"] = target
    elif target is not None:
        raise constants.ValidationError(
            "Protocol Config.execution_spec.microbenchmark_target is only allowed on microbenchmark protocols"
        )

    if spec["base_eval_args"]:
        try:
            eval_tokens = shlex.split(spec["base_eval_args"])
        except ValueError as exc:
            raise constants.ValidationError(
                f"invalid shell quoting in Protocol Config.execution_spec.base_eval_args: {exc}"
            ) from exc
        eq_prefixes = tuple(f"{f}=" for f in _RESERVED_EVAL_FLAGS)
        for tok in eval_tokens:
            if tok in _RESERVED_EVAL_FLAGS or tok.startswith(eq_prefixes):
                raise constants.ValidationError(
                    f"Protocol Config.execution_spec.base_eval_args cannot specify {tok!r} (managed by accuracy protocol)"
                )
    return out


def validate_hypothesis_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate a Hypothesis JSON object for `runtime.py hc hypothesis add`."""
    out = _validate_structure(raw, "hc-hypothesis.json", "Hypothesis")
    for k in ("target_canonical_ops", "target_relayout_edges", "evidence",
              "target_code_refs"):
        out.setdefault(k, [])

    ops = out["target_canonical_ops"]
    edges = out["target_relayout_edges"]
    for idx, edge in enumerate(edges):
        if edge["producer"] is None and edge["consumer"] is None:
            raise constants.ValidationError(
                f"Hypothesis.target_relayout_edges[{idx}] must have at least one non-null endpoint"
            )

    scope = constants.TargetScope(out["target_scope"])
    if scope == constants.TargetScope.CANONICAL_OPS:
        if not ops or edges:
            raise constants.ValidationError(
                "target_canonical_ops must be non-empty and target_relayout_edges empty when target_scope is canonical_ops"
            )
    elif scope == constants.TargetScope.REDUNDANT_RELAYOUT:
        if not edges or ops:
            raise constants.ValidationError(
                "target_relayout_edges must be non-empty and target_canonical_ops empty when target_scope is redundant_relayout"
            )
        if len(edges) > 1:
            # Bare `operator` on purpose (Spec Part 3, Hypotheses.target_relayout_edges):
            # allows a shared module (e.g. `attn_proj`) across `layer_kind`s (`dense`/`moe`).
            shared_ops: set[str] | None = None
            for edge in edges:
                edge_ops = {
                    ep["operator"]
                    for ep in (edge["producer"], edge["consumer"])
                    if ep is not None
                }
                shared_ops = (edge_ops if shared_ops is None else
                              (shared_ops & edge_ops))
            if not shared_ops:
                raise constants.ValidationError(
                    "all listed target_relayout_edges must share at least one common "
                    "operator name appearing as producer or consumer on every edge"
                )
    elif ops or edges:
        raise constants.ValidationError(
            "target_canonical_ops and target_relayout_edges must be empty when target_scope is step"
        )

    out["target_code_refs"] = [
        _safe_script_relpath(ref, f"Hypothesis.target_code_refs[{i}]")
        for i, ref in enumerate(out["target_code_refs"])
    ]
    return out
