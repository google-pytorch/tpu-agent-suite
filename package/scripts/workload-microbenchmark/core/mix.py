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
"""A job's snapshot -> its STEP MIX: the step shapes the engine actually ran, with counts.

Why not percentile points
=========================
A point `cached / input / output @ C` at the job's P50 / P90 treats every step as a batch
of C identical requests. An agentic trace is not that. GLM-5.2 AgentX at client
concurrency 4 (j-58d59d77): 54% of engine time is decode steps carrying 1-2 requests, 15%
3-4, 3% 5-8, 18% prefill chunks, 10% the client thinking between turns. Its P50 / P90
points run 7-8 requests per decode step -- 3% of that job's time, while the 1-2-request
step, the bucket-16 program, is never run. So the plan is built from the steps
themselves, each weighted by how much of the job it was.

How a step is classified (from `snapshot["step_slices"]`, `jobdata.read_step_slices`)
=====================================================================================
`vllm:iteration_tokens_total` is observed when a step ENDS, one row per second, in the
histogram's own bins (`step_slices["edges"]`). A decoding request is 1+k tokens under MTP k
(1 without), so the decode classes follow from the job (`decode_classes`): each bin whose
tokens fit mns x (1+k) requests is the class of the requests it can hold. Under MTP 3 with
mns 8 that is <=8 / <=16 / <=32 tokens = D 1-2 / 3-4 / 5-8; without MTP, <=1 / <=8 = D1 /
D2-8. A class's mean D is the number of requests in their decode phase (the snapshot's
requests) during its seconds.
  above them     a prefill chunk rides the step (mixed).
  step time      class times fitted on fully busy decode-only seconds (1 s = sum n_j t_j);
                 the rest of a second is mixed when a long step is in flight, else idle.
Validated on j-c964a8b6, which also logged every scheduler step (corrected for async
scheduling's one-step lag): mixed 74% here vs 69% from the log, D5-8 17% vs 21%, D1-2
8% vs 10%. The ~5-point error runs toward mixed.

Cells
=====
Each class is split into three context bins (0-50%, 50-90%, 90-100% of its time, each at
its time-weighted median context) and into the two integer batches bracketing its mean D.
A mixed cell's prefix is the prefilling request's cache hit under this deployment
(`cells.with_hit`), binned the same way. Every cell carries its step count and the share
of measured busy time it stands for, so a plan's result rolls up as `sum steps x t`.
"""

from __future__ import annotations

import collections
import math
import statistics
from typing import Any

from .cells import with_hit
from .deploy import Deploy
from .jobdata import SLICE_EDGES, prefix_chain

BINS = ((0.0, 0.5, "p0-50"), (0.5, 0.9, "p50-90"), (0.9, 1.0001, "p90-100"))


def decode_classes(edges: list[str],
                   d: Deploy) -> list[tuple[str, int, int, list[int]]]:
    """(name, fewest requests, most requests, histogram bins) per decode class. A bin is decode
    when it starts below mns x (1+k) tokens and ends below the step budget; a bin too narrow to
    hold a whole request (<=1 token under MTP) joins the next class."""
    rows = 1 + (d.spec_tokens if d.spec_method else 0)
    out, pending, lo = [], [], 0.0
    for i, e in enumerate(edges):
        hi = math.inf if e == "+Inf" else float(e)
        if not (lo < d.mns * rows and hi < d.mnbt):
            break
        pending.append(i)
        dlo, dhi = math.floor(lo / rows) + 1, min(math.floor(hi / rows), d.mns)
        if dhi >= dlo:
            out.append((f"D{dlo}" if dlo == dhi else f"D{dlo}-{dhi}", dlo, dhi,
                        pending))
            pending = []
        lo = hi
    return out


def _per_class(n: list[float], classes) -> list[float]:
    return [sum(n[i] for i in c[3]) for c in classes]


def _long(n: list[float], classes) -> float:
    """Steps in the bins past the decode classes: a prefill chunk rode them."""
    return sum(n[classes[-1][3][-1] + 1:]) if classes else sum(n)


def _lsq(A: list[list[float]], b: list[float]) -> dict[int, float]:
    """least squares over a few columns, dropping non-positive ones (a tiny NNLS)."""
    cols = [k for k in range(len(A[0])) if any(r[k] for r in A)]
    while cols:
        n = len(cols)
        M = [[sum(r[cols[i]] * r[cols[j]] for r in A)
              for j in range(n)] + [sum(r[cols[i]] * y for r, y in zip(A, b))]
             for i in range(n)]
        ok = True
        for i in range(n):
            p = max(range(i, n), key=lambda r: abs(M[r][i]))
            M[i], M[p] = M[p], M[i]
            if abs(M[i][i]) < 1e-12:
                ok = False
                break
            for r in range(n):
                if r != i:
                    f = M[r][i] / M[i][i]
                    M[r] = [x - f * y for x, y in zip(M[r], M[i])]
        if not ok:
            cols = cols[:-1]
            continue
        x = [M[i][n] / M[i][i] for i in range(n)]
        neg = [c for c, v in zip(cols, x) if v <= 0]
        if not neg:
            return dict(zip(cols, x))
        cols = [c for c in cols if c not in neg]
    return {}


def _slices(snap: dict) -> list[dict]:
    ss = snap.get("step_slices")
    if not ss or not ss.get("rows"):
        raise ValueError(
            "the snapshot has no step_slices (aiperf's server-metrics scrape); "
            "re-snapshot the job with the current jobdata, or pass --point")
    e = len(ss.get("edges") or SLICE_EDGES)
    return [{
        "t0": r[0],
        "n": list(r[1:1 + e]),
        "running": r[1 + e],
        "rmin": r[2 + e],
        "run_avg": r[3 + e]
    } for r in ss["rows"]]


def class_times(sl: list[dict], classes) -> tuple[dict[int, float], int]:
    """Step seconds per D class, fitted on fully busy decode-only seconds: gauge never 0,
    and no step past the decode classes in the second or its neighbours."""
    A, b = [], []
    for i in range(1, len(sl) - 1):
        if sl[i]["rmin"] > 0 and all(
                _long(x["n"], classes) == 0 and sum(x["n"]) > 0
                for x in sl[i - 1:i + 2]):
            A.append(_per_class(sl[i]["n"], classes))
            b.append(1.0)
    fit = _lsq(A, b) if A else {}
    known = list(fit.values()) or [0.05]
    t = {j: fit.get(j, statistics.median(known)) for j in range(len(classes))}
    for j in range(1, len(classes)):  # a bigger batch never runs faster
        t[j] = max(t[j], t[j - 1])
    return t, len(A)


def classify(sl: list[dict], t: dict[int, float], classes) -> None:
    """Split each second into decode class j / "mixed" / "idle" (s["f"])."""
    for i, s in enumerate(sl):
        n = s["n"]
        w = [c * t[j] for j, c in enumerate(_per_class(n, classes))]
        dec = min(sum(w), 1.0)
        f: collections.Counter = collections.Counter()
        for j in range(len(classes)):
            if w[j]:
                f[j] = dec * w[j] / sum(w)
        nxt = sl[i + 1] if i + 1 < len(sl) else None
        long_step = _long(
            n, classes) > 0 or (sum(n) == 0 and s["running"] > 0) or (
                nxt is not None and nxt["running"] > 0 and
                (_long(nxt["n"], classes) > 0 or sum(nxt["n"]) == 0))
        f["mixed" if long_step and s["running"] > 0 else "idle"] += 1 - dec
        s["f"] = f


def _wmedian(pairs: list[tuple[float, float]]) -> float:
    pairs = sorted(pairs)
    tot, acc = sum(w for _, w in pairs), 0.0
    for x, w in pairs:
        acc += w
        if acc >= tot / 2:
            return x
    return pairs[-1][0] if pairs else 0.0


def _bins(pairs: list[tuple[float, float]]) -> list[tuple[str, float, float]]:
    """(label, time share, median value) for the three bins."""
    pairs = sorted(pairs)
    tot = sum(w for _, w in pairs) or 1.0
    cur: dict[str, list] = {k: [] for _, _, k in BINS}
    acc = 0.0
    for x, w in pairs:
        mid = (acc + w / 2) / tot
        acc += w
        for a, b, k in BINS:
            if a <= mid < b:
                cur[k].append((x, w))
                break
    return [(k, sum(w for _, w in cur[k]) / tot, _wmedian(cur[k]))
            for _, _, k in BINS if cur[k]]


def _split(mean: float, lo: int, hi: int) -> list[tuple[int, float]]:
    """The two integer batches bracketing `mean`, with the shares that reproduce it."""
    mean = min(max(mean, lo), hi)
    a, b = math.floor(mean), math.ceil(mean)
    return [(a, 1.0)] if a == b else [(a, b - mean), (b, mean - a)]


def build(snap: dict, d: Deploy, cache_hit=None) -> dict[str, Any]:
    sl = _slices(snap)
    classes = decode_classes((snap.get("step_slices") or {}).get("edges")
                             or list(SLICE_EDGES), d)
    t, nfit = class_times(sl, classes)
    classify(sl, t, classes)
    rows = 1 + (d.spec_tokens if d.spec_method else 0)
    # requests, in the slices' time origin; the prefilling request's cache hit per this deploy
    hits = {
        (r["conv"], r["turn"], r["t0"]): with_hit(r, d, cache_hit)["hit"]
        for r in prefix_chain(snap["requests"])
    }
    reqs = []
    for r in snap["requests"]:
        if r.get("phase") != "profiling" or r.get("error") or not r.get("t1") \
                or r.get("ttft_ms") is None:
            continue
        ft = r["t0"] + r["ttft_ms"] / 1e3
        reqs.append({
            **r, "ft": ft,
            "hit": hits.get((r["conv"], r["turn"], r["t0"]), 0)
        })

    def ctx(q: dict, at: float) -> float:
        return q["isl"] + q["osl"] * (at - q["ft"]) / max(
            q["t1"] - q["ft"], 1e-9)

    steps = [sum(s["n"][j] for s in sl) for j in range(len(sl[0]["n"]))]
    cls_steps = _per_class(steps, classes)
    n_mixed = _long(steps, classes)
    secs: collections.Counter = collections.Counter()
    dsum: collections.Counter = collections.Counter()
    cctx: dict[Any, list] = collections.defaultdict(list)
    pre_hit, mixed_dec = [], []
    for s in sl:
        a, b, mid = s["t0"], s["t0"] + 1, s["t0"] + 0.5
        dec = [(q, min(b, q["t1"]) - max(a, q["ft"])) for q in reqs
               if q["ft"] < b and q["t1"] > a]
        nd = sum(o for _, o in dec)
        mc = sum(ctx(q, mid) * o for q, o in dec) / nd if nd else 0.0
        for k, v in s["f"].items():
            secs[k] += v
            if k == "idle" or v <= 0:
                continue
            if k == "mixed":
                pre = [q for q in reqs if q["t0"] <= mid < q["ft"]]
                if pre:
                    pre_hit.append((max(q["hit"] for q in pre), v))
                mixed_dec.append((max(0.0, s["run_avg"] - 1), v))
                if mc:
                    cctx["mixed"].append((mc, v))
            else:
                dsum[k] += nd * v
                if mc:
                    cctx[k].append((mc, v))
    busy = sum(v for k, v in secs.items() if k != "idle")
    t_mixed = secs["mixed"] / n_mixed if n_mixed else 0.0

    cells = []
    for j, (name, dlo, dhi, _) in enumerate(classes):
        if not cls_steps[j] or not secs[j]:
            continue
        mean_d = dsum[j] / secs[j]
        for blab, share, L in _bins(cctx[j]):
            for bt, bshare in _split(mean_d, dlo, dhi):
                n = cls_steps[j] * share * bshare
                cells.append({
                    "name": f"dec{name[1:]}-b{bt}-{blab}",
                    "kind": "decode",
                    "class": name,
                    "batch": bt,
                    "context": int(L),
                    "steps": round(n, 1),
                    "measured_step_ms": round(1e3 * t[j], 1),
                    "share": round(secs[j] * share * bshare / busy, 4)
                })
    if n_mixed and secs["mixed"]:
        mean_d = sum(x * w for x, w in mixed_dec) / sum(w
                                                        for _, w in mixed_dec)
        L = int(_wmedian(cctx["mixed"])) if cctx["mixed"] else 0
        for blab, share, P in _bins(pre_hit):
            for bt, bshare in _split(mean_d, 0, d.mns - 1):
                n = n_mixed * share * bshare
                chunk = d.mnbt - bt * rows
                cells.append({
                    "name":
                    f"mixed-b{bt}-{blab}",
                    "kind":
                    "mixed",
                    "class":
                    "mixed",
                    "batch":
                    bt,
                    "context":
                    L,
                    "chunk":
                    chunk,
                    "prefix":
                    int(P),
                    "steps":
                    round(n, 1),
                    "measured_step_ms":
                    round(1e3 * t_mixed, 1),
                    "share":
                    round(secs["mixed"] * share * bshare / busy, 4)
                })
    wall = sum(secs.values())
    names = {j: "decode " + c[0] for j, c in enumerate(classes)}
    return {
        "_what":
        "the steps the job's engine ran (core/mix.py): one cell per step shape, "
        "with its count and the share of measured busy time it stands for",
        "rows_per_decode":
        rows,
        "window_s":
        round(wall, 1),
        "busy_s":
        round(busy, 1),
        "idle_s":
        round(secs["idle"], 1),
        "measured_share_of_wall": {
            names.get(k, k): round(v / wall, 4)
            for k, v in sorted(secs.items(), key=str)
        },
        "measured_step_ms": {
            **{
                c[0]: round(1e3 * t[j], 1)
                for j, c in enumerate(classes)
            }, "mixed": round(1e3 * t_mixed, 1)
        },
        "steps": {
            **{
                c[0]: cls_steps[j]
                for j, c in enumerate(classes)
            }, "mixed": n_mixed
        },
        "fit_seconds":
        nfit,
        "method":
        "vllm:iteration_tokens_total 1-s slices (observed at step end) + the "
        "requests' decode intervals; validated on j-c964a8b6's step log within "
        "~5 points, biased toward mixed",
        "cells":
        cells,
    }
