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
"""Turn roofline.py `--layer-csv` runs into the roofline_data.json decompose joins on.

    python3 roofline_data.py --run <cfg>.json <cfg>.layers.csv [--run ...] -o roofline_data.json

One `--run` per config; each of its phases is one regime. `rows` is flat, one
per regime, layer kind and operator, so DuckDB reads it as a table. Every
number is modelled.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import sys

# The sibling roofline.py, whose Hardware fields are all required.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import roofline  # noqa: E402

BOUND_BY = {"HBM": "HBM", "MXU": "MXU", "VPU": "VPU", "ICI": "collective"}

# Attention-core ops roofline.py builds as raw `Op`s; their dims, read off Ctx.
RAW_SHAPES = {
    "MLA: QK Nope": lambda c: dict(batch=c.Ha, m=c.T, k=c.dc, n=c.scan),
    "MLA: QK Rope": lambda c: dict(batch=c.Ha, m=c.T, k=c.rope, n=c.scan),
    "MLA: AV": lambda c: dict(batch=c.Ha, m=c.T, k=c.scan, n=c.dc),
    "Attn: QK": lambda c: dict(batch=c.Ha, m=c.T, k=c.dh, n=c.scan),
    "Attn: AV": lambda c: dict(batch=c.Ha, m=c.T, k=c.scan, n=c.dh),
    "DSA Indexer: QK": lambda c: dict(batch=c.Hi, m=c.T, k=c.Di, n=c.Sk),
    "DSA Indexer: Weighted Sum":
    lambda c: dict(batch=1, m=c.T, k=c.Hi, n=c.Sk),
    "Attn: softmax": lambda c: dict(elems=c.T * c.Ha * c.scan),
}


def _num(x: float) -> int | float:
    return int(x) if float(x).is_integer() else x


def pass_shapes(cfg: roofline.Config,
                ph: roofline.Phase) -> dict[str, dict | None]:
    """Per op name, its local dims for one pass, checked against the op's flops.

    `gemm`/`vpu` are wrapped while `build_ops` runs, so the dims are the ones
    roofline.py priced. Collectives map to None; other ops absent from the
    result have no recoverable shape.
    """
    got, real_gemm, real_vpu = {}, roofline.gemm, roofline.vpu

    def gemm(c, name, cls, tokens, contract, out, *, n_mat=1, **kw):
        op = real_gemm(c, name, cls, tokens, contract, out, n_mat=n_mat, **kw)
        got[op.name] = dict(batch=1, m=tokens, k=contract, n=out, n_mat=n_mat)
        return op

    def vpu(c, name, cls, flops, elems, *a, **kw):
        op = real_vpu(c, name, cls, flops, elems, *a, **kw)
        got[op.name] = dict(elems=elems)
        return op

    roofline.gemm, roofline.vpu = gemm, vpu
    try:
        ops = roofline.build_ops(cfg, ph)
    finally:
        roofline.gemm, roofline.vpu = real_gemm, real_vpu
    c, out = roofline.Ctx(cfg, ph), {}
    for op in ops:
        if op.kind == "collective":
            out[op.name] = None
            continue
        s = got.get(op.name) or (RAW_SHAPES[op.name](c)
                                 if op.name in RAW_SHAPES else None)
        if s is None:
            continue
        if op.name == "Attn: softmax":
            ok = math.isclose(op.flops, 3 * s["elems"])
        elif "elems" in s:
            ok = math.isclose(op.act_bytes, s["elems"] * c.n.b("act_out"))
        else:
            s.setdefault("n_mat", 1)
            ok = math.isclose(
                2 * s["batch"] * s["m"] * max(s["k"], c.hw.mxu_min_dim) *
                s["n"] * s["n_mat"], op.flops)
        if not ok:
            raise ValueError(f"{ph.name}: {op.name}: captured dims {s} do "
                             f"not reproduce flops {op.flops}")
        out[op.name] = {
            k: _num(v)
            for k, v in s.items()
        } | {
            "dtype": op.dtype,
            "dtype_b": op.dtype_b
        }
    return out


def regime_shapes(cfg: roofline.Config,
                  ph: roofline.Phase) -> dict[str, dict | None]:
    """`pass_shapes` of the first pass; fields that change between passes are
    also listed per pass under `by_pass`."""
    per = [pass_shapes(cfg, sub) for sub in roofline.chunks_of(ph)]
    out = {}
    for name, s in per[0].items():
        if s is None:
            out[name] = None
            continue
        all_s = [p.get(name) for p in per]
        varies = [k for k in s if len({repr(x and x[k]) for x in all_s}) > 1]
        out[name] = dict(s, by_pass=[{
            k: x[k]
            for k in varies
        } for x in all_s]) if varies else s
    return out


def layer_types(model: roofline.Model) -> dict[str, dict]:
    """Which layer indices are each kind, by the rule roofline.py prices."""
    indexer = set(model.indexer_layer_ids())
    kinds: dict[str, list[int]] = {}
    for i in range(model.layers):
        kind = ("dense" if i < model.dense_layers else
                "moe") + ("_idx" if i in indexer else "")
        kinds.setdefault(kind, []).append(i)
    return {k: {"count": len(v), "layers": v} for k, v in kinds.items()}


def rows_of(path: str) -> list[dict]:
    out, parent = [], None
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            role = r["Role"]
            name = r["Operator"].removeprefix("--> ")
            if role == "fused":
                parent = name
            lat, bw = float(r["ICI Latency Time (ms)"]), float(
                r["ICI BW Time (ms)"])
            out.append({
                "regime":
                r["Phase"],
                "layer_kind":
                r["Layer Kind"],
                "count_per_step":
                int(r["Layer Count"]),
                "role":
                role,
                "operator":
                name,
                "parent":
                parent if role == "sub" else None,
                "floor_ms":
                float(r["Roofline Time (ms)"]),
                "layer_total_ms":
                float(r["Layer Total (ms)"]),
                "memory_ms":
                float(r["Memory Time (ms)"]),
                "compute_ms":
                float(r["Compute Time (ms)"]),
                "ici_ms":
                lat + bw,
                "ici_latency_ms":
                lat,
                "ici_bw_ms":
                bw,
                "bound_by":
                BOUND_BY[r["Boundness"]],
                "act_mib":
                float(r["Memory Occupancy - Activations (MiB)"]),
                "weight_mib":
                float(r["Memory Occupancy - Weights (MiB)"]),
                "kv_mib":
                float(r["Memory Occupancy - KV (MiB)"]),
                "comment":
                r["Comment"],
            })
    return out


def build(runs: list[tuple[str, str]]) -> dict:
    types, regimes, rows = None, {}, []
    for cfg_path, csv_path in runs:
        cfg = roofline.load(cfg_path)
        mine = layer_types(cfg.model)
        if types is not None and mine != types:
            raise ValueError(f"{cfg_path}: layer type map differs from the "
                             "first run's; one model per file")
        types = mine
        run_rows = rows_of(csv_path)
        for r in run_rows:
            if mine.get(r["layer_kind"],
                        {}).get("count") != r["count_per_step"]:
                raise ValueError(
                    f"{csv_path}: {r['layer_kind']} x{r['count_per_step']} "
                    f"does not match {cfg_path}'s layer map")
        phases = {p.name: p for p in cfg.phases}
        for name in dict.fromkeys(r["regime"] for r in run_rows):
            if name in regimes:
                raise ValueError(f"regime {name!r} appears in two runs")
            p = phases[name]
            passes = len(roofline.chunks_of(p))
            shapes = regime_shapes(cfg, p)
            for r in run_rows:
                if r["regime"] != name:
                    continue
                r["passes"] = passes
                r["floor_ms_per_pass"] = r["floor_ms"] / passes
                r["shape"] = shapes.get(r["operator"])
                if r["shape"] is None and r["role"] in ("op", "sub"):
                    why = ("collective: its payload is in the ici_* fields"
                           if r["operator"] in shapes else
                           "not built by gemm()/vpu() and not read off Ctx")
                    r["comment"] = "; ".join(x for x in (r["comment"],
                                                         f"shape null: {why}")
                                             if x)
            regimes[name] = {
                "config": cfg_path,
                "layer_csv": csv_path,
                "tokens": p.tokens,
                "kv_len": p.kv_len,
                "batch": p.batch,
                "mnbt": p.mnbt,
                "passes": passes,
            }
        rows += run_rows
    return {
        "provenance": "modeled",
        "layers": sum(t["count"] for t in types.values()),
        "layer_types": types,
        "regimes": regimes,
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run",
                    nargs=2,
                    action="append",
                    required=True,
                    metavar=("CONFIG", "LAYER_CSV"))
    ap.add_argument("-o", "--output", required=True)
    a = ap.parse_args(argv)
    data = build([tuple(r) for r in a.run])
    pathlib.Path(a.output).write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {a.output} ({len(data['regimes'])} regimes, "
          f"{len(data['rows'])} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
