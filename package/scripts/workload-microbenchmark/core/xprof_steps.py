#!/usr/bin/env python3
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
"""Split a twin profile into steps and layers, and project each step to the served model.

    python3 core/xprof_steps.py <trace_events.json> --plan <plan>.steps.json \\
        [--result result.json] [--json out.json]

This module does not read the trace itself. The tpu-diagnostics agent reads the rank-0 trace and
exports the events of the `XLA Modules` and `XLA Ops` lines of /device:TPU:0 to
`trace_events.json` (SKILL.md, "Reading the profile"); this module reads that export:

    {"events": [{"line": "XLA Modules" | "XLA Ops", "name": <full event name>,
                 "start_ns": <start>, "duration_ns": <duration>}, ...]}

(`line_id` / `event` / `offset_ps` / `duration_ps` are accepted for the same fields.)

Steps are the `XLA Modules` events on /device:TPU:0 (one per forward). Layers come from the
synchronous all-reduces on `XLA Ops`, counted with the model's rule carried in the plan
(`plan["trace"]`, from model.py TRACE): `embed_all_reduces` before layer 0, then
`all_reduces_per_layer` per layer. A step whose count doesn't match is reported, not split. The
rule is then tested by a per-layer op census (the model's op families per layer kind).

Per layer: ms, top ops by TensorCore time, collectives with their output shape (async ones:
the `call-done`, whose duration is the exposed wait), and the model's per-step ops found inside
it (`TRACE["per_step_ops"]`). Projection (modelled): served step ~= before + after + per-step ops
once + sum over layer kinds (real count x (twin layer ms - its per-step ops)) + drafter busy +
sampling busy. Twin collectives span the twin's ranks, not the served group; the projection does
not correct for that.

The sample stage (plans with `sampling.run`: LM head, bonus-token sampler, rejection sampler) runs
after each forward between two marker programs (`SAMPLING_MARKS`); its modules are booked as the
step's `sampling_*`, apart from the backbone and the drafter.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

_COLL = re.compile(
    r"^(all-reduce|all-gather|reduce-scatter|all-to-all|collective-permute|"
    r"distributed_\w+)")
_SYNC_AR = re.compile(
    r"^all-reduce\.\d+(?:\.cloned\.\d+\.call-done)?(?:\.refs_\d+_)*$")


def _boundary(trace: dict | None) -> re.Pattern:
    """The layer-boundary all-reduce: the model's `boundary_regex` (TRACE) if it names one, else
    either a synchronous `all-reduce.N` or the `call-done` of an async
    `all-reduce.N.cloned.M.call-start/-update/-done` triple (including XLA `.refs_k_` suffixes),
    whose end is where the layer's output is complete."""
    rx = (trace or {}).get("boundary_regex")
    return re.compile(rx) if rx else _SYNC_AR


def _short(name: str) -> str:
    return re.match(r"%?([\w.\-]+)", name).group(1)


def _kind(short: str) -> str:
    """Group key: drop buffer-donor `.refs_k_`, clone suffixes, and instance numbers."""
    s = re.sub(r"(?:\.refs_\d+_)+$", "", short)
    s = re.sub(r"\.cloned.*$", "", s)
    return re.sub(r"(\.\d+)+$", "", s)


def _shape(name: str, short: str) -> str:
    m = re.match(r"\s*=\s*(\([^=]*?\)|[a-z0-9]+\[[\d,]*\])",
                 name[name.find(short) + len(short):])
    return re.sub(r"\{[^}]*\}", "", m.group(1)) if m else ""


_ARR = re.compile(r"\b(pred|[subf]\d+(?:e\d+m\d+\w*)?|bf16)\[([\d,]*)\]")


def _shapes(name: str) -> tuple[str, list[str]]:
    """Output shape and operand shapes from an op's HLO text, layouts dropped:
    '%x = s32[2048,2048]{1,0} fusion(s32[2048,8]{..} %p, ...)' -> ('s32[2048,2048]', ['s32[2048,8]', ...])."""
    eq = name.find(" = ")
    if eq < 0:
        return "", []
    rhs = re.sub(r"\{[^{}]*\}", "",
                 name[eq + 3:])  # drop layouts: {1,0:T(8,128)} has "T("
    op = re.search(r"\)?\s*[\w\-]+\(", rhs)
    out_part, args = (rhs[:op.start()], rhs[op.end():]) if op else (rhs, "")

    def fmt(m):
        return f"{m.group(1)}[{m.group(2)}]"

    outs = [fmt(m) for m in _ARR.finditer(out_part)]
    ins = [fmt(m) for m in _ARR.finditer(args.split("), ")[0])]
    return (outs[0] if len(outs) == 1 else "(" + ", ".join(outs[:3]) +
            ")") if outs else "", ins[:4]


def _busy_ns(ops: list) -> int:
    """Device busy of a set of (start, duration, name) ops: the union of their intervals (the
    gaps between ops, e.g. host dispatch between drafter modules, are not busy)."""
    tot, end = 0, None
    for s, d, _ in sorted(ops):
        e = s + d
        if end is None or s >= end:
            tot += d
            end = e
        elif e > end:
            tot += e - end
            end = e
    return tot


def per_step_match(trace: dict | None):
    """The model's per-step ops (`TRACE["per_step_ops"]`: op-kind regex `op`, optional output-shape
    regex `out`): ops the served step runs ONCE, though the twin places them inside a layer
    (e.g. the RoPE cache copy, the attention metadata schedule). Returns name -> label or None."""
    rules = [(re.compile(r["op"]),
              re.compile(r["out"]) if r.get("out") else None,
              r.get("what", r["op"]))
             for r in (trace or {}).get("per_step_ops") or []]

    def match(name: str) -> str | None:
        sh = _short(name)
        k = _kind(sh)
        for op, out, what in rules:
            if op.search(k) and (out is None or out.search(_shapes(name)[0])):
                return what
        return None

    return match


#: forward_run.SAMPLING_MARKS: the sample stage runs between two tiny marker programs whose ops
#: output s32[start] and s32[end] (never a model tensor's shape). torch-tpu may run both sizes
#: through ONE compiled program (j-3e69c2dd: both markers ran program 2275894090417154849 and the
#: trace shows s32[7919] for both), so the trace cannot tell start from end by shape: markers are
#: read as a toggle, the first after a backbone opening the stage and the second closing it.
SAMPLING_MARKS = (7919, 7907)


def _marker(ops: list, marks: tuple[int, int] = SAMPLING_MARKS) -> str | None:
    """'mark' when a module's ops are a sampling marker program (either size), else None."""
    want = {f"s32[{m}]" for m in marks}
    for _, _, n in ops:
        if _shapes(n)[0] in want:
            return "mark"
    return None


def _assign(mods: list[tuple], is_step, marker) -> list[dict]:
    """Group a profile's modules into steps (pure, unit-tested). `mods` = [(t0, t1, name, ops)],
    `is_step(name, ops, n_steps)` says a module is a backbone forward, `marker(ops)` is truthy for
    a sampling marker program. After a backbone module, modules between its first and second
    marker are that step's SAMPLING (the markers themselves count nowhere); every other module up
    to the next backbone is its DRAFTER. A step with other than 2 markers is flagged downstream."""
    steps: list[dict] = []
    in_samp = False
    for t0, t1, mname, o in mods:
        mk = marker(o)
        if mk and steps:
            in_samp = steps[-1][
                "sampling_marks"] == 0  # first marker opens, second closes
            steps[-1]["sampling_marks"] += 1
            steps[-1]["marker_busy_ns"] += _busy_ns(
                o)  # should be ~1 us: a fused neighbour
            continue  # op would show up here
        if is_step(mname, o, len(steps)):
            in_samp = False
            steps.append({
                "module": _short(mname),
                "raw_module": mname,
                "t0": t0,
                "t1": t1,
                "ops": o,
                "draft_ns": 0,
                "draft_busy_ns": 0,
                "draft_modules": 0,
                "sampling_ns": 0,
                "sampling_busy_ns": 0,
                "sampling_modules": 0,
                "sampling_marks": 0,
                "marker_busy_ns": 0
            })
            continue
        if not steps:
            continue
        key = "sampling" if in_samp else "draft"
        steps[-1][f"{key}_ns"] += t1 - t0
        steps[-1][f"{key}_busy_ns"] += _busy_ns(o)
        steps[-1][f"{key}_modules"] += 1
    return steps


def read_events(path: str) -> tuple[list[tuple], list[tuple]]:
    """The device plane's module and op events from tpu-diagnostics's export: modules as
    (start_ns, end_ns, name) in time order, ops as (start_ns, duration_ns, name) sorted."""
    raw = json.load(open(path))
    evs = raw.get("events", raw) if isinstance(raw, dict) else raw
    mods, ops = [], []
    for e in evs:
        line = e.get("line", e.get("line_id"))
        name = e.get("name", e.get("event"))
        if "start_ns" in e:
            t0, d = int(e["start_ns"]), int(e["duration_ns"])
        else:
            t0, d = int(e["offset_ps"]) // 1000, int(e["duration_ps"]) // 1000
        if line == "XLA Modules":
            mods.append((t0, t0 + d, name))
        elif line == "XLA Ops":
            ops.append((t0, d, name))
    return sorted(mods), sorted(ops)


def load(events: str,
         backbone_ars: int | None = None,
         boundary: re.Pattern = _SYNC_AR,
         marks: tuple[int, int] = SAMPLING_MARKS) -> list[dict]:
    """The profiled steps. A step starts at each backbone module; modules after it and before
    the next backbone module are that step's drafter (plans made with --mtp), except those
    between the sampling markers, which are its sample stage (plans with `sampling.run`).

    With `backbone_ars` (the trace rule's sync all-reduce count for the twin's backbone), a
    backbone module is one with exactly that many; modules before the first are dropped (eager
    setup ops). This is needed when the drafter's forward compiles to the same module name as
    the backbone (Qwen3.5's MTP layer). Without it, the first module's name is the backbone."""
    import bisect
    mods, ops = read_events(events)
    starts = [o[0] for o in ops]

    def mine(t0, t1):
        return ops[bisect.bisect_left(starts, t0):bisect.
                   bisect_right(starts, t1)]

    backbone = _short(mods[0][2]) if mods else None

    def is_step(mname, o, n_steps):
        if backbone_ars is not None:
            return sum(1 for _, _, n in o
                       if boundary.match(_short(n))) == backbone_ars
        return _short(mname) == backbone or not n_steps

    return _assign([(t0, t1, n, mine(t0, t1)) for t0, t1, n in mods], is_step,
                   lambda o: _marker(o, marks))


def split(step: dict, n_layers: int, trace: dict | None) -> dict:
    t0, t1 = step["t0"], step["t1"]
    ars = [(s, d) for s, d, n in step["ops"]
           if _boundary(trace).match(_short(n))]
    # every bucket compiles to its own program under the same module name; the id in parentheses
    # tells them apart within one trace (not across boots: each boot assigns its own)
    prog = step.get("raw_module", "").partition("(")[2].rstrip(")") or None
    out = {
        "module": step["module"].split("(")[0],
        "program": prog,
        "ms": round((t1 - t0) / 1e6, 3),
        "all_reduces": len(ars),
        "draft_ms": round(step["draft_ns"] / 1e6, 3),
        "draft_busy_ms": round(step.get("draft_busy_ns", 0) / 1e6, 3),
        "draft_modules": step["draft_modules"],
        # the sample stage (LM head + sampler) between the markers: module span and device
        # busy (union of its ops); 0 modules = the plan did not run it
        "sampling_ms": round(step.get("sampling_ns", 0) / 1e6, 4),
        "sampling_busy_ms": round(step.get("sampling_busy_ns", 0) / 1e6, 4),
        "sampling_modules": step.get("sampling_modules", 0),
        "sampling_marks": step.get("sampling_marks", 0),
        "marker_busy_ms": round(step.get("marker_busy_ns", 0) / 1e6, 4)
    }
    if not trace:
        out["error"] = "the plan has no trace rules (model.py TRACE): not split"
        return out
    e, p = trace["embed_all_reduces"], trace["all_reduces_per_layer"]
    if len(ars) != e + p * n_layers:
        out["error"] = f"{len(ars)} sync all-reduces, expected {e + p * n_layers}: not split"
        return out

    def end(k):
        return ars[k][0] + ars[k][1]

    edges = [t0, end(e - 1) if e else t0
             ] + [end(e - 1 + p * (i + 1)) for i in range(n_layers)] + [t1]
    segs = []
    once = per_step_match(trace)
    for a, b in zip(edges, edges[1:]):
        ops = [(s, d, n) for s, d, n in step["ops"] if a <= s < b]
        ps = [(d, n, once(n)) for s, d, n in ops]
        ps = [(d, n, w) for d, n, w in ps if w]
        by: dict[str, float] = {}
        big: dict[str,
                  tuple[float,
                        str]] = {}  # kind -> (longest instance, its HLO text)
        for s, d, n in ops:
            k = _kind(_short(n))
            by[k] = by.get(k, 0.0) + d
            if d > big.get(k, (0, ""))[0]:
                big[k] = (d, n)
        coll = []
        for s, d, n in ops:
            sh = _short(n)
            sh_base = re.sub(r"(?:\.refs_\d+_)+$", "", sh)
            if _COLL.match(sh_base) and not sh_base.endswith(
                ("call-start", "call-update")):
                coll.append({
                    "t_ms": round((s - t0) / 1e6, 2),
                    "op": _kind(sh),
                    "shape": _shape(n, sh),
                    "ms": round(d / 1e6, 3)
                })
        segs.append({
            "ms":
            round((b - a) / 1e6, 3),
            # device busy: the union of the segment's op intervals (no idle inside)
            "busy_ms":
            round(_busy_ns(ops) / 1e6, 4),
            # [kind, ms, output shape, operand shapes] of the longest instance, for the
            # kinds that took time: a kind whose instances all take 0 ns (copy-start) is
            # not where the segment's time went, and has no longest instance
            "top": [[k, round(v / 1e6, 3), *_shapes(big[k][1])]
                    for k, v in sorted(by.items(), key=lambda kv: -kv[1])[:5]
                    if v > 0],
            "collectives":
            coll,
            # the model's per-step ops inside this segment (counted once, not x n_kind)
            "per_step_ms":
            round(sum(d for d, _, _ in ps) / 1e6, 4),
            "per_step_ops":
            [[w, _kind(_short(n)),
              _shapes(n)[0],
              round(d / 1e6, 4)] for d, n, w in ps if d >= 1000]
        })
    out.update({"before": segs[0], "layers": segs[1:-1], "after": segs[-1]})
    # op census per layer: "attention" = from the layer's start to the end of its first
    # all-reduce, "ffn" = the rest of the layer (census_check tests the boundary rule with it)
    for i in range(n_layers):
        lo, mid, hi = edges[i + 1], end(e + p * i), edges[i + 2]

        def kinds(a, b):
            return sorted(
                {_kind(_short(n))
                 for s, d, n in step["ops"] if a <= s < b})

        out["layers"][i]["census"] = {
            "attention": kinds(lo, mid),
            "ffn": kinds(mid, hi)
        }
    return out


def census_check(sp: dict, kinds: list[str],
                 census: dict | None) -> list[dict]:
    """Test the layer-boundary rule against what each layer contains, with the model's op
    families (`census`: family -> regex and layer part; kind -> families that must be present,
    every other family absent). A wrong boundary rule puts ops in the wrong layer and fails."""
    if not census:
        return [{
            "layer": i,
            "kind": k,
            "ok": None,
            "why": "no census in the plan"
        } for i, k in enumerate(kinds)]
    fams = {
        n: (re.compile(f["regex"]), f["part"])
        for n, f in census["families"].items()
    }
    out = []
    for i, k in enumerate(kinds):
        c = (sp.get("layers") or [{}] * len(kinds))[i].get("census")
        if not c:
            out.append({
                "layer": i,
                "kind": k,
                "ok": False,
                "why": "no census"
            })
            continue
        want = set(census["kinds"].get(k, []))
        seen = {
            n: any(rx.search(x) for x in c[part])
            for n, (rx, part) in fams.items()
        }
        out.append({
            "layer": i,
            "kind": k,
            "present": seen,
            "ok": all(seen[n] == (n in want) for n in fams)
        })
    return out


def project(sp: dict, layers: dict, trace: dict | None = None) -> dict | None:
    """Served step (modelled) = before + after + the per-step ops found inside twin layers (once)
    + sum over kinds of n_kind x (twin layer ms - its per-step ops) + the drafter's device busy
    (once: it is one layer served and in the twin; its host gaps are the replay's, not serving's).
    `per_kind_ms` is the per-layer part only, so `wall - sum(per_kind_ms)` is the fixed part."""
    if "layers" not in sp:
        return None
    raw = {k: sp["layers"][i]["ms"] for i, k in enumerate(layers["twin"])}
    once = {
        k: sp["layers"][i].get("per_step_ms", 0.0)
        for i, k in enumerate(layers["twin"])
    }
    per_kind = {k: round(raw[k] - once[k], 4) for k in raw}
    per_step = round(sum(once.values()), 4)
    body = sum(layers["real"][k] * per_kind[k] for k in layers["real"])
    draft = sp.get("draft_busy_ms") if sp.get(
        "draft_busy_ms") else sp["draft_ms"]
    samp = sp.get("sampling_busy_ms") or 0.0

    def bb(seg):
        return seg.get("busy_ms", seg["ms"])

    out = {
        "per_kind_ms":
        per_kind,
        "layer_ms_raw":
        raw,
        "per_step_in_layers_ms":
        per_step,
        "real_counts":
        layers["real"],
        "draft_ms":
        sp["draft_ms"],
        "draft_busy_ms":
        sp.get("draft_busy_ms"),
        # the sample stage's device busy, once per step (None: not run in this profile)
        "sampling_busy_ms":
        sp.get("sampling_busy_ms") if sp.get("sampling_modules") else None,
        "before_busy_ms":
        bb(sp["before"]),
        "after_busy_ms":
        bb(sp["after"]),
        "served_step_ms":
        round(
            sp["before"]["ms"] + sp["after"]["ms"] + per_step + body + draft +
            samp, 3),
        # all device time: before/after busy + per-step ops once + n_k x layer + drafter busy
        # + sampling busy (the cost model's `served_device` basis, per profile)
        "served_device_step_ms":
        round(
            bb(sp["before"]) + bb(sp["after"]) + per_step + body +
            (sp.get("draft_busy_ms") or 0.0) + samp, 4),
        "label":
        "modelled: (twin layer ms - per-step ops) x served layer counts + per-step "
        "ops once + drafter device busy once + sampling device busy once; twin-size "
        "collectives"
    }
    pair = (trace or {}).get("drafter_topk_pair")
    if sp.get("draft_modules") and pair and all(k in per_kind for k in pair):
        # A cost the drafter replay can't show (the model says which: model.py TRACE), modelled
        # as the difference of two twin layers at the same step.
        est = round(per_kind[pair[0]] - per_kind[pair[1]], 3)
        out["drafter_topk_modelled_ms"] = est
        out["drafter_topk_rule"] = f"twin {pair[0]} layer - twin {pair[1]} layer, same step"
        out["served_step_with_drafter_topk_ms"] = round(
            out["served_step_ms"] + est, 1)
    return out


def served_summary(plan: dict, steps: list[dict]) -> list[dict]:
    """Per point, from the projected mixed/decode steps (modelled): TTFT without queueing =
    prefill steps x t_mixed (a partial last chunk counted as a full one), and the steady output
    rate of C requests at this point from the plan's step mix."""
    by = {s["label"]: s for s in steps if s.get("projection")}
    out = []
    # step-mix cells: each profiled cell stands for its class (the plan profiles the heaviest
    # cell per class); served busy ~ sum over classes of the job's steps x that cell's step
    cls_ms: dict[str, tuple[str, float]] = {}
    for p in plan["points"]:
        c = p.get("cell")
        if not c:
            continue
        n = p["point"]["name"]
        st = by.get(f"{n}/mixed/chunk0") if c["kind"] == "mixed" else by.get(
            f"{n}/decode")
        if st:
            cls_ms.setdefault(c["class"],
                              (n, st["projection"]["served_step_ms"]))
    if cls_ms:
        mx = plan.get("mix") or {}
        counts = mx.get("steps") or {}
        busy = sum(counts.get(k, 0) * ms
                   for k, (_, ms) in cls_ms.items()) / 1e3
        out.append({
            "point":
            "step mix",
            "by_class": {
                k: {
                    "profiled_cell": n,
                    "served_step_ms": ms,
                    "steps": counts.get(k),
                    "measured_step_ms": (mx.get("measured_step_ms")
                                         or {}).get(k)
                }
                for k, (n, ms) in cls_ms.items()
            },
            "served_busy_s":
            round(busy, 1),
            "measured_busy_s":
            mx.get("busy_s"),
            "classes_missing":
            sorted(set(counts) - set(cls_ms)),
            "label":
            "modelled from twin profile: each class at its heaviest cell's step; "
            "no host dispatch; sampling only when the plan ran it; twin-size collectives"
        })
    for p in plan["points"]:
        if p.get("cell"):
            continue
        n = p["point"]["name"]
        mx = next((v for k, v in by.items() if k.startswith(f"{n}/mixed")),
                  None)
        dc = by.get(f"{n}/decode")
        if not (mx and dc):
            continue
        tm, td = mx["projection"]["served_step_ms"], dc["projection"][
            "served_step_ms"]
        # with the drafter's step-0 top-k (modelled) when the plan ran the drafter
        tm2 = mx["projection"].get("served_step_with_drafter_topk_ms")
        td2 = dc["projection"].get("served_step_with_drafter_topk_ms")
        mix = p["step_mix"]
        T = mix["mixed_steps"] * tm + mix["decode_steps"] * td
        chunks = p["chunks_full"] + (1 if p["last_chunk_tokens"] else 0)
        C, osl = p["point"]["concurrency"], p["point"]["output"]
        extra = {
            "t_mixed_with_drafter_topk_ms": tm2,
            "t_decode_with_drafter_topk_ms": td2,
            "ttft_no_queue_with_drafter_topk_s": round(chunks * tm2 / 1e3, 2)
        } if tm2 else {}
        out.append({
            "point":
            n,
            "t_mixed_ms":
            tm,
            "t_decode_ms":
            td,
            **extra, "ttft_no_queue_s":
            round(chunks * tm / 1e3, 2),
            "output_tok_per_s":
            round(C * osl / (T / 1e3), 1) if T else None,
            "label":
            "modelled from twin profile" + ("; drafter included" if any(
                x.get("draft_modules") for x in steps) else "; no drafter") +
            "; no host dispatch; sampling only when the plan ran it; twin-size collectives"
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("events",
                    help="trace_events.json, exported by tpu-diagnostics")
    ap.add_argument("--plan", required=True)
    ap.add_argument(
        "--result",
        help="forward_run result.json: names the steps (profile_order)")
    ap.add_argument("--json")
    a = ap.parse_args()
    plan = json.load(open(a.plan))
    layers = plan["layers"]
    trace = plan.get("trace")
    order = (json.load(open(a.result)).get("profile_order")
             or []) if a.result else []
    ars = (trace["embed_all_reduces"] +
           trace["all_reduces_per_layer"] * len(layers["twin"])
           if trace and trace.get("backbone_by_all_reduces") else None)
    steps = load(a.events, ars, _boundary(trace))
    if order and len(order) != len(steps):
        print(
            f"! {len(steps)} XLA modules in the trace, {len(order)} steps in profile_order",
            file=sys.stderr)
    out = []
    for i, st in enumerate(steps):
        sp = split(st, len(layers["twin"]), trace)
        sp["label"] = order[i]["label"] if i < len(order) else f"step{i}"
        sp["wall_ms"] = order[i]["ms"] if i < len(order) else None
        if i < len(order) and order[i].get("sampling") and sp.get(
                "sampling_marks", 0) < 2:
            # the stage ran but its markers were not found: its modules sit in the drafter's
            sp["sampling_error"] = (
                f"{sp.get('sampling_marks', 0)} of 2 sampling markers found; "
                "sampling modules are counted as drafter")
        # where to find this step in Trace Viewer: ms after the first profiled step starts
        sp["start_ms"] = round((st["t0"] - steps[0]["t0"]) / 1e6, 2)
        sp["projection"] = project(sp, layers, trace)
        if "layers" in sp:
            sp["census_check"] = census_check(sp, layers["twin"],
                                              (trace or {}).get("census"))
        out.append(sp)
        print(
            f"{sp['label']:22} device {sp['ms']:8.2f} ms (wall {sp['wall_ms']})"
            +
            (f"  ! {sp['error']}" if "error" in sp else
             f"  layers {[layer['ms'] for layer in sp['layers']]}  before {sp['before']['ms']} "
             f"after {sp['after']['ms']}" +
             (f"  draft {sp['draft_ms']} ({sp['draft_modules']} modules)"
              if sp['draft_modules'] else "") +
             (f"  sampling busy {sp['sampling_busy_ms']} ({sp['sampling_modules']} modules)"
              if sp.get('sampling_modules') else "") +
             f"  -> served ~{sp['projection']['served_step_ms']} ms"))
        for j, layer in enumerate(sp.get("layers", [])):
            print(
                f"    L{j} {layers['twin'][j]:9} {layer['ms']:8.2f}  top {layer['top'][:3]}"
            )
    served = served_summary(plan, out)
    for r in served:
        if "by_class" in r:
            print(
                f"step mix served (modelled): busy {r['served_busy_s']} s vs measured "
                f"{r['measured_busy_s']} s; " + ", ".join(
                    f"{k} {v['served_step_ms']} ms (measured {v['measured_step_ms']})"
                    for k, v in r["by_class"].items()) +
                (f"; not profiled: {r['classes_missing']}"
                 if r["classes_missing"] else ""))
            continue
        print(
            f"{r['point']:5} served (modelled): mixed {r['t_mixed_ms']} ms, decode "
            f"{r['t_decode_ms']} ms, TTFT w/o queueing {r['ttft_no_queue_s']} s, "
            f"{r['output_tok_per_s']} output tok/s")
    j = plan.get("job_measured") or {}
    if j:
        print(
            f"job {j.get('job_id')} (measured): mixed {j.get('t_mixed_s')} s, ITL {j.get('itl_ms')} ms, "
            f"TTFT {j.get('ttft_ms')} ms, {j.get('output_tok_per_s')} output tok/s"
        )
    if a.json:
        json.dump({
            "steps": out,
            "served": served,
            "job_measured": j
        },
                  open(a.json, "w"),
                  indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
