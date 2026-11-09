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
"""RewardKit verifier embedding the original deterministic grade.py engine."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from rewardkit import criterion


def _coerce_number(v: Any) -> float | None:
    """Accepts 99.8, '99.8', or tool format '99.8%'."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip().rstrip("%").replace(",", "")
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _flatten(d: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out = {}
    for k, v in (d or {}).items():
        path = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, path + "."))
        else:
            out[path] = v
    return out


def _resolve_fields(schema: dict[str, Any]) -> dict[str, Any]:
    """Expands '*.field' wildcard specs across groups."""
    groups = schema.get("groups") or []
    resolved = {}
    for pattern, spec in schema.get("fields", {}).items():
        if pattern.startswith("*."):
            suffix = pattern[2:]
            if not groups:
                raise ValueError(
                    f"field pattern {pattern!r} needs a non-empty `groups`")
            for g in groups:
                resolved[f"{g}.{suffix}"] = spec
        else:
            resolved[pattern] = spec
    return resolved


def _check_field(path: str, spec: dict[str, Any], expected: Any,
                 got: Any) -> tuple[bool, dict[str, Any]]:
    """Checks a single field against expected value according to schema type and tolerances."""
    weight = spec.get("weight", 1)
    base = {"field": path, "weight": weight, "expected": expected, "got": got}

    if got is None:
        return False, {**base, "ok": False, "why": "missing"}

    kind = spec.get("type")

    if kind in ("enum", "string"):
        if isinstance(expected, (list, tuple, set)):
            acceptable = [str(x).strip() for x in expected]
            if isinstance(got, (list, tuple, set)):
                ok = any(str(x).strip() in acceptable for x in got)
            else:
                ok = str(got).strip() in acceptable
        else:
            if isinstance(got, (list, tuple, set)):
                ok = str(expected).strip() in [str(x).strip() for x in got]
            else:
                ok = str(got).strip() == str(expected).strip()

        why = None if ok else "mismatch"
        if kind == "enum":
            values = spec.get("values", [])
            if isinstance(got, (list, tuple, set)):
                if not all(str(x).strip() in values for x in got):
                    why = "not in vocabulary"
            elif str(got).strip() not in values:
                why = "not in vocabulary"
        return ok, {**base, "ok": ok, **({"why": why} if why else {})}

    if kind == "bool":
        ok = bool(got) is bool(expected)
        return ok, {**base, "ok": ok, **({} if ok else {"why": "mismatch"})}

    if kind == "set":
        try:
            ok = set(map(str, got)) == set(map(str, expected))
        except TypeError:
            return False, {**base, "ok": False, "why": "not a list"}
        return ok, {
            **base, "ok": ok,
            **({} if ok else {
                   "why": "set mismatch"
               })
        }

    if kind == "number":
        g, e = _coerce_number(got), _coerce_number(expected)
        if g is None:
            return False, {**base, "ok": False, "why": "not numeric"}
        if "min_value" in spec:
            min_v = float(spec["min_value"])
            ok = g >= min_v
            return ok, {
                **base,
                "ok": ok,
                "min_value": min_v,
                **({} if ok else {
                       "why": f"value {g} below minimum {min_v}"
                   }),
            }

        tol_abs = spec.get("tol_abs", 0.0)
        tol_rel = spec.get("tol_rel", 0.0)
        allowed = max(tol_abs, abs(e) * tol_rel)
        delta = abs(g - e)
        ok = delta <= allowed
        return ok, {
            **base,
            "ok": ok,
            "delta": round(delta, 6),
            "allowed": round(allowed, 6),
            **({} if ok else {
                   "why": "out of tolerance"
               }),
        }

    raise ValueError(f"unknown field type {kind!r} for {path}")


def _check_constraints(schema: dict[str, Any],
                       answer: dict[str, Any]) -> list[dict[str, Any]]:
    """Evaluates structural constraints across groups (e.g. distinct values)."""
    results = []
    groups = schema.get("groups") or []
    for c in schema.get("constraints", []):
        if c.get("kind") == "distinct":
            vals = [(answer.get(g) or {}).get(c["field"]) for g in groups]
            present = [v for v in vals if v is not None]
            ok = len(set(map(str, present))) == len(groups)
            results.append({
                "constraint":
                "distinct",
                "field":
                c["field"],
                "ok":
                ok,
                "values":
                vals,
                **({} if ok else {
                       "why": c.get("message", "values not distinct")
                   }),
            })
    return results


def grade(
    task_dir: Path,
    answer_path: Path,
    schema_path: Path | None = None,
    key_path: Path | None = None,
) -> dict[str, Any]:
    """Original grade() implementation from cli_mini_eval."""
    search_dirs = [task_dir]
    if task_dir.parent not in search_dirs:
        search_dirs.append(task_dir.parent)
    # We expect answer and schema to be mounted at root test directory.
    if Path("/tests") not in search_dirs and Path("/tests").is_dir():
        search_dirs.append(Path("/tests"))

    if not schema_path or not schema_path.is_file():
        for d in search_dirs:
            for cand in (
                    d / "schema.json",
                    d / "answer.schema.json",
            ):
                if cand.is_file():
                    schema_path = cand
                    break
            if schema_path and schema_path.is_file():
                break
            schema_cands = list(d.glob("*.schema.json"))
            if schema_cands:
                schema_path = schema_cands[0]
                break

    if not schema_path or not schema_path.is_file():
        raise FileNotFoundError(
            f"no schema found under {[str(d) for d in search_dirs]} (expected 'schema.json' or"
            " '*.schema.json')")

    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    if not key_path or not key_path.is_file():
        for d in search_dirs:
            for cand in (
                    d / "answer.json",
                    d / "key.json",
                    d / "answer_key.json",
                    d / "key" / "answer.json",
                    d / "key" / "answer_key.json",
            ):
                if cand.is_file():
                    key_path = cand
                    break
            if key_path and key_path.is_file():
                break

    if not key_path or not key_path.is_file():
        raise FileNotFoundError(
            f"no answer key found under {[str(d) for d in search_dirs]} (expected 'answer.json',"
            " 'key.json', or 'answer_key.json')")

    key = json.loads(key_path.read_text(encoding="utf-8"))

    if not answer_path.is_file():
        return {
            "task":
            schema.get("task", "unknown"),
            "passed":
            False,
            "score":
            0,
            "max":
            sum(s.get("weight", 1) for s in _resolve_fields(schema).values()),
            "status":
            "NO_ANSWER_FILE",
            "checks": [],
        }

    try:
        answer = json.loads(answer_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {
            "task":
            schema.get("task", "unknown"),
            "passed":
            False,
            "score":
            0,
            "max":
            sum(s.get("weight", 1) for s in _resolve_fields(schema).values()),
            "status":
            f"MALFORMED_JSON: {e}",
            "checks": [],
        }

    fields = _resolve_fields(schema)
    flat_key, flat_ans = _flatten(key), _flatten(answer)

    checks, score, total = [], 0, 0
    for path, spec in fields.items():
        if path not in flat_key:
            continue
        weight = spec.get("weight", 1)
        total += weight
        ok, detail = _check_field(path, spec, flat_key[path],
                                  flat_ans.get(path))
        if ok:
            score += weight
        checks.append(detail)

    constraints = _check_constraints(schema, answer)
    threshold = schema.get("pass_threshold", total)
    passed = score >= threshold and all(c["ok"] for c in constraints)

    return {
        "task": schema.get("task", "unknown"),
        "passed": passed,
        "score": score,
        "max": total,
        "pass_threshold": threshold,
        "status": "GRADED",
        "constraints": constraints,
        "checks": checks,
    }


def _discover_paths() -> tuple[Path, Path | None, Path | None]:
    """Discovers task directory and optional explicit schema/key paths from injected args or flags."""
    args_list = globals().get("_SPECIFIED_ARGS", sys.argv)
    if not isinstance(args_list, (list, tuple)):
        args_list = sys.argv

    task_dir = Path(__file__).resolve().parent
    schema_path: Path | None = None
    key_path: Path | None = None

    for i in range(len(args_list) - 1):
        arg = str(args_list[i])
        if arg in ("--task-dir", "-t", "--task"):
            task_dir = Path(args_list[i + 1])
        elif arg in ("--schema", "-s"):
            schema_path = Path(args_list[i + 1])
        elif arg in ("--key", "-k", "--answer-key"):
            key_path = Path(args_list[i + 1])

    if schema_path and not schema_path.is_absolute():
        schema_path = task_dir / schema_path
    if key_path and not key_path.is_absolute():
        key_path = task_dir / key_path

    return task_dir, schema_path, key_path


@criterion(description="Verify task answer against schema and answer key")
def check_task_grade(workspace: Path) -> bool:
    task_dir, schema_path, key_path = _discover_paths()
    answer_file = workspace / "answer.json"

    print(
        f"check_task_grade: checking '{answer_file}' using config in '{task_dir}'"
    )

    if not answer_file.is_file():
        print("check_task_grade: fail (answer.json not found in workspace)")
        return False

    try:
        result = grade(
            task_dir=task_dir,
            answer_path=answer_file,
            schema_path=schema_path,
            key_path=key_path,
        )
    except Exception as e:
        print(f"check_task_grade: error ({e})")
        return False

    task_id = result.get("task", "task")
    score = result.get("score", 0)
    max_score = result.get("max", 0)
    threshold = result.get("pass_threshold", max_score)
    passed = result.get("passed", False)

    print(f"check_task_grade: {task_id} score={score}/{max_score} "
          f"(threshold={threshold}) -> {'pass' if passed else 'fail'}")

    # Output specific failures for clear scorecard visibility
    if not passed:
        for c in result.get("checks", []):
            if not c.get("ok"):
                print(
                    f"  [FAIL] {c.get('field')}: expected={c.get('expected')!r} got={c.get('got')!r}"
                )
        for c in result.get("constraints", []):
            if not c.get("ok"):
                print(
                    f"  [FAIL] constraint:{c.get('constraint')} ({c.get('why')})"
                )

    return passed
