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
"""Single-host TPU pytest microbenchmark generator and `RESULT` gate parser.

Before building a container image or launching a full-engine
`workload-microbenchmark` A/B for a kernel size-limit or tile change (such as
extending `ROW_BUCKETS = (16, 32, 64)` to `80`), verify compile time, chained
`jax.lax.scan` net microseconds/layer (`438 us -> 175 us` per layer, `2.50x`
speedup on `tpu7x`), and float64 reference numerics (`rel_l2 < 0.06`,
`cos > 0.998`) in isolation on a single TPU host.
"""

import dataclasses
import json
import re
import textwrap
from collections.abc import Mapping, Sequence
from typing import Any

from hillclimb import code_limits

DEFAULT_CANDIDATE_PATHS = ("candidate", "kernel")

# Matches both bare `RESULT {...}` lines and `pytest -s` lines prefixed by a
# test node id (`tests/.../test_moe.py::test_rows[80] RESULT {...}`), while
# rejecting indented traceback source lines (`print("RESULT " + ...)`).
_RESULT_LINE_RE = re.compile(r"^\s*(?:\S+::[^\n]*?\s+)?RESULT (\{.*\})\s*$")


@dataclasses.dataclass(frozen=True)
class BenchSpec:
    """Parameters for a generated single-host TPU pytest microbenchmark."""

    name: str
    module: str
    limit_name: str
    limit_kind: str
    base_values: tuple[int, ...]
    steps: tuple[int, ...]
    hidden_size: int = 6144
    env: Mapping[str, str] = dataclasses.field(default_factory=dict)
    layers: int = 32
    reps: int = 10
    max_rel_l2: float = 0.06
    min_cos: float = 0.998
    entry_point: str = ""


@dataclasses.dataclass(frozen=True)
class StepVerdict:
    """Speed and numerics gate outcome at one row bucket."""

    rows: int
    status: str  # pass | slower | numerics_fail | admitted_false | error
    candidate_path: str
    candidate_net_us: float | None = None
    fallback_path: str | None = None
    fallback_net_us: float | None = None
    speedup: float | None = None
    rel_l2: float | None = None
    cos: float | None = None
    compile_s: float | None = None
    detail: str = ""


@dataclasses.dataclass(frozen=True)
class BenchVerdict:
    """Aggregate microbenchmark gate verdict across all evaluated step rows."""

    passed: bool
    device_kind: str
    devices: int
    chain_overhead_us: float
    steps: tuple[StepVerdict, ...]
    kernel: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def module_from_path(rel_path: str) -> str:
    """Converts `src/pkg/mod.py` -> `pkg.mod`."""
    cleaned = rel_path.strip().replace("\\", "/")
    if cleaned.startswith("src/"):
        cleaned = cleaned[len("src/"):]
    if cleaned.endswith("/__init__.py"):
        cleaned = cleaned[:-len("/__init__.py")]
    elif cleaned.endswith(".py"):
        cleaned = cleaned[:-len(".py")]
    return cleaned.replace("/", ".")


def spec_from_crossing(
    crossing: code_limits.Crossing,
    base_steps: Sequence[int] = (),
    env: Mapping[str, str] | None = None,
    hidden_size: int = 6144,
    entry_point: str = "",
) -> BenchSpec:
    """Builds a `BenchSpec` from a `code_limits.Crossing`."""
    module = module_from_path(crossing.limit.path)
    leaf = module.rsplit(".", 1)[-1]
    name = f"{leaf}_{crossing.limit.name.lower()}"
    all_steps = tuple(sorted({*base_steps, *crossing.steps}))
    return BenchSpec(
        name=name,
        module=module,
        limit_name=crossing.limit.name,
        limit_kind=crossing.limit.kind,
        base_values=crossing.value,
        steps=all_steps,
        hidden_size=hidden_size,
        env=dict(env or {}),
        entry_point=entry_point,
    )


def extended_value(spec: BenchSpec, step: int) -> tuple[int, ...] | int:
    """Returns the limit value that admits `step`."""
    if spec.limit_kind == "table":
        return tuple(sorted({*spec.base_values, step}))
    if spec.limit_kind == "min":
        return min(spec.base_values[0], step)
    return max(spec.base_values[0], step)


def excluded_value(spec: BenchSpec, step: int) -> tuple[int, ...] | int:
    """Returns a limit value that forces the fallback path at `step`."""
    if spec.limit_kind == "table":
        return tuple(v for v in spec.base_values if v != step)
    if spec.limit_kind == "min":
        return step + 1
    return max(0, step - 1)


def render_pytest(spec: BenchSpec) -> str:
    """Emits a self-contained single-host pytest + CLI TPU microbenchmark."""
    env_json = json.dumps(dict(sorted(spec.env.items())), indent=4)
    env_literal = textwrap.indent(env_json, " " * 8).lstrip()
    base_repr = (repr(spec.base_values)
                 if spec.limit_kind == "table" else repr(spec.base_values[0]))
    call_expr = (f"target_mod.{spec.entry_point}(x, weight)"
                 if spec.entry_point else
                 "x @ weight.T  # Replace with kernel call.")
    header = (f'"""Single-host TPU microbenchmark for '
              f'{spec.module}.{spec.limit_name}."""')
    return (textwrap.dedent(f"""\
        # Copyright 2026 Google LLC
        # SPDX-License-Identifier: Apache-2.0
        {header}

        import importlib
        import json
        import os
        from pathlib import Path
        import subprocess
        import sys
        import time

        import pytest

        _ENV = {env_literal}
        ROWS = {spec.steps!r}
        HIDDEN = {spec.hidden_size}
        LAYERS = {spec.layers}
        REPS = {spec.reps}
        LIMIT_KIND = {spec.limit_kind!r}
        LIMIT_NAME = {spec.limit_name!r}
        BASE_LIMIT = {base_repr}
        TOLERANCE = {{"rel_l2": {spec.max_rel_l2}, "cos": {spec.min_cos}}}


        def _candidate_limit(rows):
            if LIMIT_KIND == "table":
                return tuple(sorted({{*BASE_LIMIT, rows}}))
            if LIMIT_KIND == "min":
                return min(BASE_LIMIT, rows)
            return max(BASE_LIMIT, rows)


        def _fallback_limit(rows):
            if LIMIT_KIND == "table":
                return tuple(v for v in BASE_LIMIT if v != rows)
            if LIMIT_KIND == "min":
                return rows + 1
            return max(0, rows - 1)


        def _bench():
            import jax
            from jax import lax
            import jax.numpy as jnp
            from jax.sharding import Mesh, NamedSharding
            from jax.sharding import PartitionSpec as P
            import numpy as np

            here = Path(__file__).resolve()
            for parent in here.parents:
                if (parent / "src" / "vllm_torchtpu").is_dir():
                    sys.path.insert(0, str(parent / "src"))
                    break
            target_mod = importlib.import_module({spec.module!r})

            devices = jax.devices()
            n_dev = len(devices)
            mesh = Mesh(np.asarray(devices), ("d",))
            key_x, key_w = jax.random.split(jax.random.key(0))
            w_host = (
                jax.random.normal(key_w, (HIDDEN, HIDDEN), jnp.float32)
                / np.sqrt(HIDDEN)
            ).astype(jnp.bfloat16)
            w = jax.device_put(w_host, NamedSharding(mesh, P()))
            saved_limit = getattr(target_mod, LIMIT_NAME, BASE_LIMIT)

            def op_body(x, weight):
                return {call_expr}

            def chained(body):
                def run(x, weight):
                    def step(carry, _):
                        y = body(carry, weight).astype(jnp.float32)
                        y = y * lax.rsqrt(jnp.mean(y * y) + 1e-6)
                        return y.astype(jnp.bfloat16), None
                    out, _ = lax.scan(step, x, None, length=LAYERS)
                    return out
                return run

            def sharded(fn):
                return jax.jit(
                    jax.shard_map(
                        fn,
                        mesh=mesh,
                        in_specs=(P("d"), P()),
                        out_specs=P("d"),
                        check_vma=False,
                    )
                )

            def time_chain(fn, x_sharded):
                t0 = time.monotonic()
                out = fn(x_sharded, w)
                out.block_until_ready()
                compile_s = time.monotonic() - t0
                fn(x_sharded, w).block_until_ready()
                t0 = time.perf_counter()
                for _ in range(REPS):
                    out = fn(x_sharded, w)
                out.block_until_ready()
                per_layer_us = (time.perf_counter() - t0) / REPS / LAYERS * 1e6
                finite = bool(np.isfinite(np.asarray(out, np.float32)).all())
                return round(per_layer_us, 2), round(compile_s, 2), finite

            def check_numerics(out, x_host, rows, n_valid):
                arr = np.asarray(out, np.float32).reshape(n_dev, rows, -1)
                bits = arr.view(np.uint32)
                with jax.default_matmul_precision("highest"):
                    ref = np.asarray(
                        x_host[:n_valid].astype(jnp.float32)
                        @ w_host.astype(jnp.float32).T,
                        dtype=np.float64,
                    )
                got = arr[0, :n_valid].astype(np.float64)
                denom = max(float(np.linalg.norm(ref)), 1e-12)
                rel = float(np.linalg.norm(got - ref) / denom)
                cos_denom = max(float(np.linalg.norm(got)) * denom, 1e-12)
                cos = float(np.vdot(got, ref) / cos_denom)
                return {{
                    "rel_l2": round(rel, 5),
                    "cos": round(cos, 6),
                    "ranks_identical": bool((bits == bits[0]).all()),
                    "nan": int(np.isnan(arr).sum()),
                }}

            x0 = jax.device_put(
                jnp.zeros((n_dev * ROWS[0], HIDDEN), jnp.bfloat16),
                NamedSharding(mesh, P("d")),
            )
            overhead_us, _, _ = time_chain(
                sharded(chained(lambda x, _: x)), x0
            )
            result = {{
                "devices": n_dev,
                "device_kind": devices[0].device_kind,
                "jax": jax.__version__,
                "module": {spec.module!r},
                "limit_name": LIMIT_NAME,
                "layers": LAYERS,
                "reps": REPS,
                "chain_overhead_us": overhead_us,
                "cases": {{}},
            }}

            for rows in ROWS:
                n_valid = max(1, rows - 3)
                x_host = jax.random.normal(key_x, (rows, HIDDEN), jnp.bfloat16)
                x_put = jax.device_put(
                    jnp.tile(x_host, (n_dev, 1)), NamedSharding(mesh, P("d"))
                )
                for label, lim_val in (
                    ("candidate", _candidate_limit(rows)),
                    ("baseline", _fallback_limit(rows)),
                ):
                    key = f"{{rows}}:{{label}}"
                    setattr(target_mod, LIMIT_NAME, lim_val)
                    try:
                        out = sharded(op_body)(x_put, w)
                        case = check_numerics(out, x_host, rows, n_valid)
                        us, comp_s, finite = time_chain(
                            sharded(chained(op_body)), x_put
                        )
                        case.update(
                            per_layer_us=us, compile_s=comp_s, finite=finite
                        )
                        result["cases"][key] = case
                    except Exception as exc:
                        result["cases"][key] = {{
                            "error": f"{{type(exc).__name__}}: {{exc}}"[:600]
                        }}
                    finally:
                        setattr(target_mod, LIMIT_NAME, saved_limit)
            print("RESULT " + json.dumps(result), flush=True)


        def test_{spec.name}_by_rows():
            proc = subprocess.run(
                [sys.executable, __file__, "bench"],
                env=dict(os.environ, **_ENV),
                capture_output=True,
                text=True,
                timeout=2400,
                check=False,
            )
            lines = [
                ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")
            ]
            if proc.returncode != 0 and "needs a TPU host" in proc.stdout:
                pytest.skip("needs a TPU host")
            assert proc.returncode == 0 and lines, proc.stderr[-4000:]
            payload = json.loads(lines[-1][len("RESULT ") :])
            for key, case in sorted(payload["cases"].items()):
                assert "error" not in case, (key, case)
                if "rel_l2" in case:
                    assert case["rel_l2"] < TOLERANCE["rel_l2"], (key, case)
                    assert case["cos"] > TOLERANCE["cos"], (key, case)
                    assert case["ranks_identical"] and not case["nan"], (key, case)


        if __name__ == "__main__" and sys.argv[1:] == ["bench"]:
            import jax

            if jax.default_backend() != "tpu":
                print("needs a TPU host")
                sys.exit(3)
            _bench()
        """).strip() + "\n")


def parse_results(stdout: str) -> list[dict[str, Any]]:
    """Extracts all `RESULT {...}` JSON payloads (supporting `pytest -s` prefixes)."""
    raw_jsons: list[str] = []
    for line in stdout.splitlines():
        match = _RESULT_LINE_RE.match(line)
        if match is not None:
            raw_jsons.append(match.group(1))
    if not raw_jsons:
        raise ValueError("No RESULT JSON line found in benchmark output.")
    payloads: list[dict[str, Any]] = []
    for raw in raw_jsons:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as err:
            raise ValueError(f"Invalid RESULT JSON: {err}") from err
        if not isinstance(payload, dict) or not isinstance(
                payload.get("cases"), dict):
            raise ValueError(
                'RESULT JSON must be an object containing a "cases" map.')
        payloads.append(payload)
    return payloads


def parse_result(stdout: str) -> dict[str, Any]:
    """Extracts the final `RESULT {...}` JSON payload from benchmark stdout."""
    return parse_results(stdout)[-1]


def _group_cases(
    cases: Mapping[str, Mapping[str, Any]],
) -> dict[int, dict[str, Mapping[str, Any]]]:
    grouped: dict[int, dict[str, Mapping[str, Any]]] = {}
    for key, metrics in cases.items():
        if ":" not in key:
            continue
        row_str, path = key.split(":", 1)
        if not row_str.isdigit():
            continue
        grouped.setdefault(int(row_str), {})[path] = metrics
    return grouped


def _pick_candidate_path(
    by_rows: Mapping[int, Mapping[str, Mapping[str, Any]]],
    requested: str,
) -> str:
    if requested:
        return requested
    all_paths: set[str] = set()
    for row_map in by_rows.values():
        all_paths.update(row_map)
    for name in DEFAULT_CANDIDATE_PATHS:
        if name in all_paths:
            return name
    return DEFAULT_CANDIDATE_PATHS[0]


def _evaluate_step(
    rows: int,
    row_cases: Mapping[str, Mapping[str, Any]],
    candidate_path: str,
    overhead_us: float,
    max_rel_l2: float,
    min_cos: float,
) -> StepVerdict:
    cand = row_cases.get(candidate_path)
    if cand is None:
        return StepVerdict(
            rows=rows,
            status="error",
            candidate_path=candidate_path,
            detail=f"missing {rows}:{candidate_path}",
        )
    if "error" in cand:
        return StepVerdict(
            rows=rows,
            status="error",
            candidate_path=candidate_path,
            detail=str(cand["error"]),
        )
    if not cand.get("admitted", True):
        return StepVerdict(
            rows=rows,
            status="admitted_false",
            candidate_path=candidate_path,
            detail="kernel admission check returned False",
        )

    rel_l2 = float(cand["rel_l2"]) if cand.get("rel_l2") is not None else None
    cos = float(cand["cos"]) if cand.get("cos") is not None else None
    compile_s = (float(cand["compile_s"])
                 if cand.get("compile_s") is not None else None)
    cand_net = (round(max(0.0,
                          float(cand["per_layer_us"]) - overhead_us), 2)
                if cand.get("per_layer_us") is not None else None)

    best_fb_path: str | None = None
    best_fb_net: float | None = None
    for path, fb in sorted(row_cases.items()):
        if path == candidate_path or fb.get("per_layer_us") is None:
            continue
        if "error" in fb or not fb.get("finite", True):
            continue
        net = round(max(0.0, float(fb["per_layer_us"]) - overhead_us), 2)
        if best_fb_net is None or net < best_fb_net:
            best_fb_net = net
            best_fb_path = path

    speedup = (round(best_fb_net / cand_net, 3) if cand_net and cand_net > 0
               and best_fb_net is not None else None)

    failures: list[str] = []
    if not cand.get("finite", True) or int(cand.get("nan", 0)) > 0:
        failures.append("non-finite output")
    if not cand.get("ranks_identical", True):
        failures.append("cross-rank mismatch")
    if not cand.get("padding_rows_positive_zero", True):
        failures.append("non-zero padding rows")
    if rel_l2 is not None and rel_l2 >= max_rel_l2:
        failures.append(f"rel_l2={rel_l2:.5f} >= {max_rel_l2}")
    if cos is not None and cos <= min_cos:
        failures.append(f"cos={cos:.6f} <= {min_cos}")

    if failures:
        status = "numerics_fail"
        detail = "; ".join(failures)
    elif speedup is not None and speedup < 1.0:
        status = "slower"
        detail = (f"{cand_net:.1f} us > {best_fb_path} {best_fb_net:.1f} us "
                  f"({speedup:.2f}x)")
    else:
        status = "pass"
        detail = ""

    return StepVerdict(
        rows=rows,
        status=status,
        candidate_path=candidate_path,
        candidate_net_us=cand_net,
        fallback_path=best_fb_path,
        fallback_net_us=best_fb_net,
        speedup=speedup,
        rel_l2=rel_l2,
        cos=cos,
        compile_s=compile_s,
        detail=detail,
    )


def evaluate_result(
    payload: Mapping[str, Any],
    steps: Sequence[int] = (),
    candidate_path: str = "",
    max_rel_l2: float = 0.06,
    min_cos: float = 0.998,
) -> BenchVerdict:
    """Evaluates a parsed `RESULT` dictionary across target row counts."""
    by_rows = _group_cases(payload.get("cases", {}))
    chosen_path = _pick_candidate_path(by_rows, candidate_path)
    overhead_us = float(payload.get("chain_overhead_us", 0.0))
    target_rows = sorted(steps) if steps else sorted(by_rows)
    verdicts = tuple(
        _evaluate_step(
            rows,
            by_rows.get(rows, {}),
            chosen_path,
            overhead_us,
            max_rel_l2,
            min_cos,
        ) for rows in target_rows)
    passed = bool(verdicts) and all(v.status == "pass" for v in verdicts)
    kernel = str(payload.get("kernel") or payload.get("module") or "")
    return BenchVerdict(
        passed=passed,
        device_kind=str(payload.get("device_kind", "unknown")),
        devices=int(payload.get("devices", 0)),
        chain_overhead_us=overhead_us,
        steps=verdicts,
        kernel=kernel,
    )


def render_verdict(verdict: BenchVerdict) -> str:
    """Formats a `BenchVerdict` as a Markdown table."""
    gate = "PASS" if verdict.passed else "FAIL"
    title = (f"Microbenchmark gate ({verdict.kernel})"
             if verdict.kernel else "Microbenchmark gate")
    lines = [
        (f"{title}: **{gate}** ({verdict.device_kind}, "
         f"{verdict.devices} cores, chain overhead "
         f"{verdict.chain_overhead_us:.2f} us/layer)"),
        "",
        ("| Rows | Status | Candidate net (us) | Fastest fallback (us) | "
         "Speedup | rel_l2 | cos | Notes |"),
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for step in verdict.steps:
        cand = (f"{step.candidate_net_us:.1f}"
                if step.candidate_net_us is not None else "—")
        fb = (f"{step.fallback_net_us:.1f} ({step.fallback_path})"
              if step.fallback_net_us is not None else "—")
        spd = f"{step.speedup:.2f}x" if step.speedup is not None else "—"
        rel = f"{step.rel_l2:.5f}" if step.rel_l2 is not None else "—"
        cos = f"{step.cos:.6f}" if step.cos is not None else "—"
        lines.append(
            f"| {step.rows} | {step.status} | {cand} | {fb} | {spd} | "
            f"{rel} | {cos} | {step.detail or 'ok'} |")
    return "\n".join(lines) + "\n"
