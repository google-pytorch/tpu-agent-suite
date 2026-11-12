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
"""Checks agents against their contracts.

A contract is the schema a delegation to an agent must match (`accepts`) and
the result the agent owes back (`produces`). This backs the `validate-schema`
builtin hook and `runtime.py validate`.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Any

from framework import jsonio, paths, utils

Path = pathlib.Path
TpuAgentSuiteError = utils.TpuAgentSuiteError

# Turns the `validate-schema` builtin into a no-op, so every delegation request
# and subagent deliverable is allowed through unchecked.
# Note: Hook dispatch executes the built copy in the installed package
# directory (e.g. ~/.tpu-agent-suite/packages/<package_tag>/framework/ or the
# staged dist). Editing this constant in the source tree applies to future
# package builds; to disable verification in an already-built or installed
# environment via the constant, edit the built agent_contracts.py directly.
# Alternatively, set the TPU_AGENT_SUITE_DISABLE_SCHEMA_VERIFIERS environment
# variable to disable verification for any existing install without editing
# files or rebuilding.
DISABLE_SCHEMA_VERIFIERS = False
DISABLE_SCHEMA_VERIFIERS_ENV_VAR = "TPU_AGENT_SUITE_DISABLE_SCHEMA_VERIFIERS"


def validate_json_file(schema_path: Path, instance_path: Path) -> list[str]:
    """Validate a JSON instance against a JSON schema file."""
    instance = jsonio.load_json(instance_path)
    schema = jsonio.load_json(schema_path)
    errors = utils.validate_instance(instance, schema)
    if not errors and "artifacts" in schema.get("properties", {}):
        errors = check_completed_has_artifact(instance)
    if not errors:
        errors = check_status_payload(schema, instance, schema_path.name)
    return errors


def check_completed_has_artifact(instance: Any) -> list[str]:
    """Require a `completed` result to include at least one artifact entry."""
    if not isinstance(instance, dict):
        return []
    if instance.get("status") != "completed":
        return []
    artifacts = instance.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return [
            "$: a completed result must carry at least one artifact;"
            " write the output under the run's artifacts directory"
        ]
    return []


def check_status_payload(schema: dict[str, Any],
                         instance: Any,
                         schema_name: str = "") -> list[str]:
    """Check that `instance` has non-empty values for fields listed in `schema['status_requires'][status]`."""
    if not isinstance(schema, dict) or not isinstance(instance, dict):
        return []
    rules = schema.get("status_requires")
    if not isinstance(rules, dict):
        return []
    expected = rules.get(str(instance.get("status")))
    if not expected or not isinstance(expected, list):
        return []

    missing = [
        field for field in expected
        if field not in instance or instance[field] in (None, "", [], {})
    ]
    if missing:
        label = str(schema.get("$id") or schema_name or "result")
        status = str(instance.get("status"))
        return [
            f"$: a {label} reporting '{status}' must carry"
            f" {', '.join(missing)}; report the status for work that did not"
            " reach an answer, with a failure_reason, rather than claiming"
            f" '{status}' having measured nothing"
        ]
    return []


def _resolve_agent_contract(
    package: str,
    workspace: Path,
    agent_name: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Resolve `(accepts_schema, produces_schema)` for `agent_name`, or `(None, None)`."""
    if not agent_name:
        return None, None

    try:
        package_root = paths.get_installed_package_root(package)
    except TpuAgentSuiteError as exc:
        print(f"runtime: contract validation skipped -- {exc}",
              file=sys.stderr)
        return None, None

    build_file = package_root / "tpu-agent-suite-build.json"
    if not build_file.is_file():
        return None, None

    try:
        build_info = jsonio.load_json(build_file)
    except Exception:
        return None, None

    contracts = build_info.get("contracts", {})
    contract = contracts.get(agent_name)
    if not contract and ":" in agent_name:
        contract = contracts.get(agent_name.split(":", 1)[1])
    if not contract:
        return None, None

    def _load_schema(rel_path: str) -> dict[str, Any] | None:
        if not rel_path:
            return None
        candidate = package_root / rel_path
        if candidate.is_file():
            try:
                return jsonio.load_json(candidate)
            except Exception:
                return None
        return None

    accepts_schema = _load_schema(contract.get("accepts", ""))
    produces_schema = _load_schema(contract.get("produces", ""))
    return accepts_schema, produces_schema


def _delegate_routing_keys() -> set[str]:
    """Return delegation routing/metadata argument keys declared in `tool-mappings.json`."""
    keys: set[str] = set()
    try:
        blocks = utils.get_tool_mappings().get("delegate",
                                               {}).get("harnesses", {})
    except TpuAgentSuiteError:
        return keys

    for h_info in blocks.values():
        for meta_key in h_info.get("metadata", []):
            if isinstance(meta_key, str) and meta_key:
                keys.add(meta_key)
        for canonical_key, raw_spec in h_info.get("args", {}).items():
            keys.add(canonical_key)
            specs = raw_spec if isinstance(raw_spec, list) else [raw_spec]
            for spec in specs:
                if isinstance(spec, str) and spec:
                    keys.add(spec.split(".", 1)[0])
    return keys


def _delegation_denial_reason(
    subagent_name: str,
    errors: list[str],
    accepts_schema: dict[str, Any],
    ctx_run_id: str = "",
) -> str:
    """Format an `accepts` schema denial message, including `ctx_run_id` and schema example."""
    reason = (f"Delegation payload to '{subagent_name}' violated accepts"
              " schema: " + "; ".join(errors))

    if ctx_run_id and any("run_id" in err for err in errors):
        reason += (f"\n\nThis session's run_id is {ctx_run_id}; send that"
                   " value. Do not copy the run_id shown in the example.")

    example = utils.load_schema_example(accepts_schema)
    if example is not None:
        reason += ("\n\nExpected payload schema example:\n```json\n"
                   f"{json.dumps(example, indent=2)}\n```")
    return reason


def _schema_verifiers_disabled() -> bool:
    """Whether schema verification is turned off for this run."""
    return DISABLE_SCHEMA_VERIFIERS or bool(
        os.environ.get(DISABLE_SCHEMA_VERIFIERS_ENV_VAR, "").strip())


def builtin_validate_schema(
    package: str,
    workspace: Path,
    event_payload: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    """Builtin hook handler to deterministically validate payload contracts and deliverables."""
    if not event_payload or _schema_verifiers_disabled():
        return {"decision": "allow"}

    event_type = event_payload.get("event", "")
    subagent_name = str(event_payload.get("subagent") or "")

    accepts_schema, produces_schema = _resolve_agent_contract(
        package, workspace, subagent_name)

    if event_type == "PreToolUse":
        tool_name = event_payload.get("tool", "")
        if tool_name != "delegate" or not accepts_schema:
            return {"decision": "allow"}

        args = event_payload.get("args", {})
        prompt_val = args.get("prompt")
        schema_props = accepts_schema.get("properties", {})
        candidates = utils.extract_all_json_objects(prompt_val)

        routing_keys = {"subagent"}
        ctx_run_id = event_payload.get("context", {}).get("run_id")

        if candidates:
            # Check candidates in reverse order: the last JSON that satisfies
            # the schema wins.
            val_errors: list[str] = []
            matched = False
            for cand in reversed(candidates):
                payload_to_test = {
                    k: v
                    for k, v in cand.items()
                    if k not in routing_keys or k in schema_props
                }
                cand_errors = utils.validate_instance(payload_to_test,
                                                      accepts_schema)
                if not cand_errors:
                    matched = True
                    break
                if not val_errors:
                    val_errors = cand_errors

            if not matched:
                err_msg = _delegation_denial_reason(subagent_name, val_errors,
                                                    accepts_schema, ctx_run_id)
                return {"decision": "deny", "reason": err_msg}
        else:
            if "prompt" in args or "run_id" in args:
                payload_to_validate = dict(args)
                routing_keys = _delegate_routing_keys()
            else:
                payload_to_validate = {"prompt": prompt_val or ""}
                routing_keys = set()

            payload_to_validate = {
                k: v
                for k, v in payload_to_validate.items()
                if k not in routing_keys or k in schema_props
            }

            val_errors = utils.validate_instance(payload_to_validate,
                                                 accepts_schema)
            if val_errors:
                err_msg = _delegation_denial_reason(subagent_name, val_errors,
                                                    accepts_schema, ctx_run_id)
                return {"decision": "deny", "reason": err_msg}

    elif event_type == "SubagentStop":
        if not produces_schema:
            return {"decision": "allow"}

        raw_out = event_payload.get("output")
        candidates = utils.extract_all_json_objects(raw_out)
        if not candidates:
            err_msg = (
                f"Subagent '{subagent_name}' output must be a valid JSON object conforming to its produces schema."
            )
            example = utils.load_schema_example(produces_schema)
            if example is not None:
                err_msg += f"\n\nExpected output schema example:\n```json\n{json.dumps(example, indent=2)}\n```"
            return {
                "decision": "deny",
                "reason": err_msg,
                "followup_message": err_msg
            }

        # Check candidates in reverse order: the last JSON that satisfies the
        # schema wins.
        all_errors: list[str] = []
        for cand in reversed(candidates):
            val_errors = utils.validate_instance(cand, produces_schema)
            artifact_errors = (check_completed_has_artifact(cand)
                               if "artifacts" in produces_schema.get(
                                   "properties", {}) else [])
            payload_errors = check_status_payload(produces_schema, cand)
            cand_errors = list(val_errors) + list(artifact_errors) + list(
                payload_errors)
            if not cand_errors:
                return {"decision": "allow"}
            if not all_errors:
                # Save the errors of the last candidate for reporting if none
                # pass
                all_errors = cand_errors

        err_msg = (
            f"Subagent '{subagent_name}' output violated produces contract: " +
            "; ".join(all_errors))
        example = utils.load_schema_example(produces_schema)
        if example is not None:
            err_msg += f"\n\nExpected output schema example:\n```json\n{json.dumps(example, indent=2)}\n```"
        return {
            "decision": "deny",
            "reason": err_msg,
            "followup_message": err_msg
        }

    return {"decision": "allow"}
