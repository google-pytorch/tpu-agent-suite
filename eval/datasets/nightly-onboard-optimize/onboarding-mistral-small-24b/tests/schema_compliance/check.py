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
"""schema_compliance: fraction of runtime schema checks that passed.

Reads the `schema_checks.jsonl` ledger emitted at runtime by
`framework.agent_contracts.builtin_validate_schema` on `PreToolUse` (`accepts`)
and `SubagentStop` (`produces`), writes a summary to
`$LOGS_DIR/verifier/schema_compliance.json`, and returns the compliance ratio
in [0.0, 1.0] (or 1.0 when no contracted checks occurred).
"""

from __future__ import annotations

import json
import os
import pathlib
from typing import Any

from rewardkit import criterion

_LEDGER_FILENAME = "schema_checks.jsonl"
_SUMMARY = "verifier/schema_compliance.json"


def load_checks(ledger: pathlib.Path) -> list[dict[str, Any]]:
    """Read schema check records from a `schema_checks.jsonl` file."""
    checks: list[dict[str, Any]] = []
    try:
        lines = ledger.read_text(encoding="utf-8").splitlines()
    except OSError:
        return checks

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(entry, dict):
            continue
        direction = entry.get("direction")
        subagent = entry.get("subagent")
        if direction not in ("accepts", "produces") or not isinstance(
                subagent, str) or not subagent:
            continue
        raw_errors = entry.get("errors")
        errors = ([str(e) for e in raw_errors]
                  if isinstance(raw_errors, list) else [])
        compliant = bool(entry.get("compliant", not errors))
        checks.append({
            "subagent": subagent,
            "direction": direction,
            "compliant": compliant,
            "errors": errors,
        })
    return checks


def summarize(checks: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate schema check records into a summary report."""
    by_direction: dict[str, dict[str, int]] = {
        "accepts": {
            "checked": 0,
            "compliant": 0
        },
        "produces": {
            "checked": 0,
            "compliant": 0
        },
    }
    by_agent: dict[str, dict[str, int]] = {}
    failures: list[dict[str, Any]] = []

    for check in checks:
        direction = check["direction"]
        subagent = check["subagent"]
        compliant = check["compliant"]

        by_direction[direction]["checked"] += 1
        if compliant:
            by_direction[direction]["compliant"] += 1

        slot = by_agent.setdefault(
            subagent,
            {
                "accepts_checked": 0,
                "accepts_compliant": 0,
                "produces_checked": 0,
                "produces_compliant": 0,
            },
        )
        slot[f"{direction}_checked"] += 1
        if compliant:
            slot[f"{direction}_compliant"] += 1
        else:
            failures.append({
                "subagent": subagent,
                "direction": direction,
                "errors": check["errors"],
            })

    checked = len(checks)
    compliant_total = sum(1 for c in checks if c["compliant"])
    return {
        "score": compliant_total / checked if checked else None,
        "checked": checked,
        "compliant": compliant_total,
        "by_direction": by_direction,
        "by_agent": by_agent,
        "failures": failures,
    }


@criterion(description="Fraction of TPU agent suite subagent contract "
           "schema checks that passed")
def schema_compliance(workspace: pathlib.Path) -> float:
    """RewardKit script check entrypoint."""
    del workspace
    logs_dir = pathlib.Path(os.environ.get("LOGS_DIR", "/logs"))
    ledger = logs_dir / "artifacts" / _LEDGER_FILENAME
    checks = load_checks(ledger) if ledger.is_file() else []
    summary = summarize(checks)

    target = logs_dir / _SUMMARY
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, indent=2) + "\n",
                          encoding="utf-8")
    except OSError as error:
        print(f"schema_compliance: could not write {target}: {error}")

    score = summary["score"]
    if score is None:
        print("schema_compliance: no contracted schema checks recorded -> 1.0")
        return 1.0
    print(f"schema_compliance: {summary['compliant']}/{summary['checked']}"
          f" checks compliant -> {score:.4f}")
    return float(score)
