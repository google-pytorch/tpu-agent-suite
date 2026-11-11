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
"""Static source scanner for hardcoded kernel row/bucket limits in served code.

Raising `DECODE_BATCH_CAP` from 16 to 20 with `MTP k=3` pushes a full decode
step from `64` to `80` tokens, silently crossing `ROW_BUCKETS = (16, 32, 64)` in
the fused TP MoE decode kernel (its unfused GMM+psum fallback read `-28%`
Mtok/$ 11 minutes into one run).
`scan` and `new_crossings` surface every hardcoded size table, scalar row cap,
or env threshold in a checkout that an arm's step sizes cross.
"""

import ast
import dataclasses
import pathlib
from collections.abc import Mapping, Sequence

_SIZE_WORDS = ("ROW", "ROWS", "BUCKET", "BUCKETS", "TOKEN", "TOKENS", "M")
_MAX_WORDS = ("MAX", "LIMIT")
_MIN_WORDS = ("MIN", )
_NOISE_WORDS = ("PORT", "TIMEOUT", "SEED", "VERSION", "RANK", "STEP", "LAYER")

SKIP: frozenset[str] = frozenset({
    "MAX_SAFE_TOKENS_PER_CHUNK",
    "HOST_CONVERSION_CHUNK_TOKENS",
})


@dataclasses.dataclass(frozen=True)
class Limit:
    """One hardcoded size table, scalar row cap, or env threshold in source."""

    path: str
    line: int
    name: str
    kind: str  # table | max | min
    values: tuple[int, ...]
    overridable: bool = False


@dataclasses.dataclass(frozen=True)
class Crossing:
    """A `Limit` crossed by one or more candidate step token counts."""

    limit: Limit
    steps: tuple[int, ...]
    value: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.limit.path,
            "line": self.limit.line,
            "name": self.limit.name,
            "kind": self.limit.kind,
            "overridable": self.limit.overridable,
            "steps": list(self.steps),
            "value": list(self.value),
        }


def _name_tokens(name: str) -> set[str]:
    return {tok for tok in name.upper().split("_") if tok}


def _is_size_name(name: str) -> bool:
    if name in SKIP or not name.isupper() and not (name.startswith("_")
                                                   and name[1:].isupper()):
        return False
    toks = _name_tokens(name)
    if any(w in toks for w in _NOISE_WORDS):
        return False
    return any(w in toks for w in _SIZE_WORDS)


def _int_const(node: ast.AST) -> int | None:
    if (isinstance(node, ast.Constant) and not isinstance(node.value, bool)
            and isinstance(node.value, int)):
        return int(node.value)
    return None


def _env_default(node: ast.AST) -> tuple[str, int] | None:
    """Extracts `(ENV_NAME, int_default)` from `int(os.getenv('X', '64'))`."""
    inner = node
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
            and node.func.id == "int" and len(node.args) == 1):
        inner = node.args[0]
    if not isinstance(inner, ast.Call) or len(inner.args) < 2:
        return None
    fn_name = ""
    if isinstance(inner.func, ast.Attribute):
        fn_name = inner.func.attr
    elif isinstance(inner.func, ast.Name):
        fn_name = inner.func.id
    if fn_name not in ("getenv", "get"):
        return None
    k_node, v_node = inner.args[0], inner.args[1]
    if not (isinstance(k_node, ast.Constant)
            and isinstance(k_node.value, str)):
        return None
    env_name = k_node.value
    val = _int_const(v_node)
    if val is None and isinstance(v_node, ast.Constant) and isinstance(
            v_node.value, str):
        if v_node.value.strip().isdigit():
            val = int(v_node.value.strip())
    if val is None:
        return None
    return env_name, val


def scan_text(text: str, path: str = "<memory>") -> list[Limit]:
    """Parses module-level size tables, scalar row caps, and env thresholds."""
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError:
        return []

    found: list[Limit] = []
    for stmt in tree.body:
        target_name: str | None = None
        value_node: ast.AST | None = None
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
            if isinstance(stmt.targets[0], ast.Name):
                target_name = stmt.targets[0].id
                value_node = stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(
                stmt.target, ast.Name):
            target_name = stmt.target.id
            value_node = stmt.value
        if target_name is None or value_node is None:
            continue

        # 1. Tuple / list / set table of >=2 positive step ints, e.g. ROW_BUCKETS.
        if isinstance(value_node, (ast.Tuple, ast.List, ast.Set)):
            ints = [_int_const(el) for el in value_node.elts]
            if (len(ints) >= 2
                    and all(v is not None and 1 <= v <= 4096 for v in ints)
                    and _is_size_name(target_name)):
                clean_vals = tuple(
                    sorted({int(v)
                            for v in ints if v is not None}))
                if len(clean_vals) >= 2:
                    found.append(
                        Limit(
                            path=path,
                            line=stmt.lineno,
                            name=target_name,
                            kind="table",
                            values=clean_vals,
                            overridable=False,
                        ))
                continue

        # 2. Env-backed threshold: `FOO = int(os.getenv("TPU_MAX_ROWS", "64"))`.
        env_def = _env_default(value_node)
        if env_def is not None:
            env_name, env_val = env_def
            if _is_size_name(env_name) and 1 <= env_val <= 4096:
                toks = _name_tokens(env_name)
                kind = "min" if any(w in toks for w in _MIN_WORDS) else "max"
                found.append(
                    Limit(
                        path=path,
                        line=stmt.lineno,
                        name=env_name,
                        kind=kind,
                        values=(env_val, ),
                        overridable=True,
                    ))
            continue

        # 3. Direct scalar constant: `MAX_DECODE_ROWS = 64` or `_LIMIT_M = 64`.
        scalar = _int_const(value_node)
        if scalar is not None and 1 <= scalar <= 4096 and _is_size_name(
                target_name):
            toks = _name_tokens(target_name)
            if any(w in toks for w in _MIN_WORDS):
                kind = "min"
            elif any(w in toks for w in _MAX_WORDS):
                kind = "max"
            else:
                continue
            found.append(
                Limit(
                    path=path,
                    line=stmt.lineno,
                    name=target_name,
                    kind=kind,
                    values=(scalar, ),
                    overridable=False,
                ))
    return found


def scan(repo: str | pathlib.Path) -> list[Limit]:
    """Scans Python source files in `repo` (preferring `src/` if present)."""
    root = pathlib.Path(repo)
    if root.is_file():
        return scan_text(root.read_text(encoding="utf-8", errors="replace"),
                         root.name)
    scan_root = root / "src" if (root / "src").is_dir() else root
    limits: list[Limit] = []
    for py_path in sorted(scan_root.rglob("*.py")):
        rel = str(py_path.relative_to(root)).replace("\\", "/")
        if "/tests/" in f"/{rel}" or rel.endswith("_test.py"):
            continue
        limits.extend(
            scan_text(py_path.read_text(encoding="utf-8", errors="replace"),
                      rel))
    return limits


def _effective_values(
    limit: Limit,
    env: Mapping[str, str] | None,
) -> tuple[int, ...]:
    if limit.overridable and env and limit.name in env:
        raw = str(env[limit.name]).strip()
        parts = [
            int(tok.strip()) for tok in raw.split(",")
            if tok.strip().isdigit()
        ]
        if parts:
            return tuple(sorted(
                set(parts))) if limit.kind == "table" else (parts[0], )
    return limit.values


def _misses(limit_kind: str, eff_values: tuple[int, ...], step: int) -> bool:
    if limit_kind == "table":
        return step not in eff_values
    if limit_kind == "max":
        return step > eff_values[0]
    if limit_kind == "min":
        return step < eff_values[0]
    return False


def crossings(
    limits: Sequence[Limit],
    steps: Sequence[int],
    env: Mapping[str, str] | None = None,
) -> list[Crossing]:
    """Returns all `limits` missed by at least one step in `steps`."""
    unique_steps = sorted({int(s) for s in steps if int(s) > 0})
    out: list[Crossing] = []
    for lim in limits:
        eff = _effective_values(lim, env)
        crossed = tuple(s for s in unique_steps if _misses(lim.kind, eff, s))
        if crossed:
            out.append(Crossing(limit=lim, steps=crossed, value=eff))
    return out


def new_crossings(
    limits: Sequence[Limit],
    steps: Sequence[int],
    base_steps: Sequence[int],
    env: Mapping[str, str] | None = None,
    base_env: Mapping[str, str] | None = None,
) -> list[Crossing]:
    """Returns limits crossed by `steps` that were NOT crossed by `base_steps`."""
    eff_base_env = base_env if base_env is not None else env
    base_missed = {(c.limit.path, c.limit.line, c.limit.name)
                   for c in crossings(limits, base_steps, eff_base_env)}
    return [
        c for c in crossings(limits, steps, env)
        if (c.limit.path, c.limit.line, c.limit.name) not in base_missed
    ]


def render(found: Sequence[Crossing], repo: str = "") -> str:
    """Renders crossed limits as a Markdown table."""
    if not found:
        return "No hardcoded size limits crossed at the requested step sizes.\n"
    title = f"### Crossed Kernel & Env Size Limits (`{repo}`)" if repo else "### Crossed Kernel & Env Size Limits"
    lines = [
        title,
        "",
        "| File:Line | Symbol | Kind | Active Limit | Crossed Steps | Overridable |",
        "|---|---|---|---|---|---|",
    ]
    for c in found:
        val_str = (repr(c.value)
                   if c.limit.kind == "table" else str(c.value[0]))
        steps_str = ", ".join(str(s) for s in c.steps)
        ov = "env" if c.limit.overridable else "code-only"
        lines.append(
            f"| `{c.limit.path}:{c.limit.line}` | `{c.limit.name}` | "
            f"`{c.limit.kind}` | `{val_str}` | `{steps_str}` | {ov} |")
    return "\n".join(lines) + "\n"
