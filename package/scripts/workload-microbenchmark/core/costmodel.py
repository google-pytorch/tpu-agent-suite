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
"""The step-mix cost model: per-cell twin step times -> one workload number, and an A/B.

    T_mix = sum over cells i of  steps_i x t_i

`steps_i` is the cell's step count in the source job (`core/mix.py`); `t_i` is the cell's
measured twin step time (median over repeats). T_mix is the job's engine-busy time at the twin's
shapes. An A/B compares two runs of the SAME cells:

    speedup = T_mix(baseline) / T_mix(candidate)

Two bases. `twin`: t_i is the twin's wall step as run, the backbone plus (when the plan ran the
MTP drafter) the drafter's replay wall. `served` (MODELLED, the default when profiles exist): the
twin has one layer per kind, so its wall step under-weights per-layer work (a saving inside
layers is n_layers x larger in serving) and over-weights per-step work (host dispatch,
embedding, step entry and exit). `served` rebuilds each cell's step from its profile:

    t_i_served = fixed_i + sum over kinds k of  n_k x layer_k,i  + drafter_i
    fixed_i    = backbone_wall_i - sum_k layer_k,i     (host dispatch + before/after + the
                                                        model's per-step ops, counted once)
    drafter_i  = the drafter's DEVICE BUSY per step from the profile, counted once

with layer_k,i from `xprof_steps.json` (a twin layer minus the per-step ops the model lists in
TRACE["per_step_ops"], which the served step runs once although the twin places them inside a
layer) and n_k the served layer counts (from the model, `projection.real_counts`).

Why the drafter's device busy and not its wall: the drafter is one layer served and in the
twin, so it is never multiplied by n_k; but its replay (`forward_run._run_draft`) is driven from
the host module by module, and its wall (3.9-4.8 ms on Qwen3.5) is mostly host gaps a served
drafter does not have (served span 1.49 ms, busy 1.29; twin busy 1.13-1.16). Each cell keeps the
replay wall (`draft_wall_ms`) beside the busy it used (`draft_busy_ms`) and says which
(`drafter_basis`). A profile without drafter busy falls back to the wall, labelled.

A cell with no profile of its own is projected through the profiled cell of its class: its
layer body is its backbone wall minus that cell's `fixed`, scaled by that cell's served/twin
layer ratio, plus that cell's drafter busy. Each cell records which profile it used.

The sample stage (plans with `sampling.run`: LM head, bonus-token sampler, rejection sampler)
is counted like the drafter: its device busy from the profile, once per step (`sampling_busy_ms`);
its wall (`sampling_wall_ms`) is in the twin basis only.

A third basis, `served_device` (MODELLED), takes every term from device time:

    t_i_device = before + after busy + per-step ops + sum_k n_k x layer_k,i + drafter busy
                 + sampling busy

It leaves out host dispatch, so it is the basis for a change that moves device time (a kernel,
a fusion, a layout, a collective): `served`'s `fixed` comes from the host wall and carries its
per-step jitter, which on Qwen3.5 hid a -1.4% layer-time kernel win. A change aimed at host time
(dispatch, scheduling glue) is invisible here and needs `served` or `twin`. `significance` tests
two arms on their per-profile T_mix samples (Welch's t).

Everything here is arithmetic on numbers `forward_run` measured, `xprof_steps` split and `mix.py`
counted; no model facts. SKILL.md "Cost model" states what the number means and when it is valid.
"""

from __future__ import annotations

import statistics
from typing import Any


def cell_times(result_points: list[dict]) -> dict[str, dict[str, Any]]:
    """From one forward_run result's `points`: per cell, its step (median over repeats) and
    each repeat's value. A mixed cell's step is its mixed batch; a decode cell's, its decode.
    When the plan ran the MTP drafter, its propose replay (timed right after the step, `draft_ms`)
    is part of the twin step as run: median_ms = backbone wall + drafter wall, per repeat. The
    backbone alone is kept (`backbone_ms`, `backbone_by_repeat`) for the served basis, which
    takes the drafter's device busy from the profile instead of its replay wall."""
    out = {}

    def per_rep(x: dict | None, n: int) -> list[float]:
        if not x:
            return [0.0] * n
        v = list(x.get("by_repeat") or [])
        return v if len(v) == n else [x["median_after_first"]] * n

    for p in result_points:
        tm = p.get("twin_model") or {}
        if "step_ms" not in tm or tm["step_ms"] is None:
            continue
        m = p["mixed_ms"][0] if p.get("mixed_ms") else p.get("decode_ms")
        bb = list(m.get("by_repeat") or [m["median_after_first"]])
        d, sm = m.get("draft_ms"), m.get("sampling_ms")
        # the step as run: backbone wall + the sample stage's wall (plans with sampling.run) +
        # the drafter replay's wall (plans with --mtp), per repeat
        reps = [
            b + x + y
            for b, x, y in zip(bb, per_rep(d, len(bb)), per_rep(sm, len(bb)))
        ]
        med = m["median_after_first"] + (d["median_after_first"] if d else 0.0) \
            + (sm["median_after_first"] if sm else 0.0)
        out[p["point"]["name"]] = {
            "median_ms": round(med, 4),
            "by_repeat": [round(x, 4) for x in reps],
            "backbone_ms": m["median_after_first"],
            "backbone_by_repeat": bb,
            "draft_ms": d["median_after_first"] if d else None,
            "sampling_wall_ms": sm["median_after_first"] if sm else None
        }
    return out


def cell_profiles(xsteps: list[dict],
                  cells: dict[str, dict],
                  into: dict[str, list] | None = None,
                  job: str = "") -> dict[str, list[dict[str, Any]]]:
    """From one job's `xprof_steps.json` steps: per cell, the profiled step that IS the cell (a
    mixed cell's first chunk, a decode cell's decode), split into twin layer ms and served counts.
    Profiles of the same cell from several jobs of one arm pool into a list (pass `into`): one
    profile is one step, and a per-layer wobble is multiplied by the served layer count."""
    out = into if into is not None else {}
    for st in xsteps:
        name, _, rest = (st.get("label") or "").partition("/")
        pr = st.get("projection")
        if name not in cells or not pr or not pr.get("per_kind_ms"):
            continue
        if rest != ("mixed/chunk0"
                    if cells[name]["kind"] == "mixed" else "decode"):
            continue
        out.setdefault(name, []).append({
            "per_kind_ms":
            pr["per_kind_ms"],
            "served_counts":
            pr["real_counts"],
            "job":
            job,
            # drafter device busy after this step (None: no drafter
            # in the profile, or an older xprof_steps.json)
            "draft_busy_ms":
            pr.get("draft_busy_ms") or None,
            # the sample stage's device busy (None: not run)
            "sampling_busy_ms":
            pr.get("sampling_busy_ms"),
            # the rest of the step's device time (served_device)
            "before_busy_ms":
            pr.get("before_busy_ms"),
            "after_busy_ms":
            pr.get("after_busy_ms"),
            "per_step_ms":
            pr.get("per_step_in_layers_ms", 0.0)
        })
    return out


def _by_job(samples: list[dict]) -> list[list[dict]]:
    groups: dict[str, list[dict]] = {}
    for x in samples:
        groups.setdefault(x.get("job", ""), []).append(x)
    return list(groups.values())


def _pooled(samples: list[dict], pick: int | None = None) -> dict[str, float]:
    """Per layer kind: the median within each job (boot), then the median over jobs; or one
    job's median (`pick`, by job order). Returns twin and served layer-body ms."""
    jobs = _by_job(samples)
    if pick is not None:
        jobs = [jobs[min(pick, len(jobs) - 1)]]
    kinds = samples[0]["per_kind_ms"]
    per = {
        k:
        statistics.median(
            statistics.median(x["per_kind_ms"][k] for x in g) for g in jobs)
        for k in kinds
    }
    cnt = samples[0]["served_counts"]
    dj = [[x["draft_busy_ms"] for x in g if x.get("draft_busy_ms")]
          for g in jobs]
    draft = (statistics.median(statistics.median(d)
                               for d in dj) if dj and all(dj) else None)
    sj = [[
        x["sampling_busy_ms"] for x in g
        if x.get("sampling_busy_ms") is not None
    ] for g in jobs]
    samp = (statistics.median(statistics.median(d)
                              for d in sj) if sj and all(sj) else None)
    return {
        "layers_twin_ms": sum(per.values()),
        "layers_served_ms": sum(cnt[k] * per[k] for k in cnt),
        "per_kind_ms": per,
        "draft_busy_ms": draft,
        "sampling_busy_ms": samp
    }


def served_times(cells: dict[str, dict],
                 times: dict[str, dict],
                 profiles: dict[str, list],
                 pick: int | None = None) -> dict[str, dict[str, Any]]:
    """`times` (twin wall) -> the same shape at the served layer counts (MODELLED). A cell uses its
    own profiles, else its class's profiled cell with the largest share; with neither it is left
    out (and listed missing by `evaluate`). `pick` uses only profile #pick of each cell (to
    measure profile-to-profile spread); default the per-kind median over all of them."""
    out = {}
    for n, t in times.items():
        if n not in cells:
            continue
        src = n if n in profiles else max(
            (m for m in profiles if m in cells and m in times
             and cells[m]["class"] == cells[n]["class"]),
            key=lambda m: cells[m]["share"],
            default=None)
        if src is None:
            continue
        pf = _pooled(profiles[src], pick)
        busy = pf["draft_busy_ms"]
        # the drafter: its device busy from the profile, once; else (no profiled busy) its
        # replay wall stays in the step as run, labelled
        use_busy = t.get("draft_ms") is not None and times[src].get("draft_ms") is not None \
            and busy is not None
        samp = pf["sampling_busy_ms"]
        # the sample stage: its device busy from the profile, once per step (like the drafter);
        # its wall stays out of `fixed` (fixed is the backbone wall minus its layers)
        use_samp = samp is not None and t.get("sampling_wall_ms") is not None \
            and times[src].get("sampling_wall_ms") is not None
        if use_busy:
            w_src, w_med = times[src]["backbone_ms"], t["backbone_ms"]
            w_reps = t.get("backbone_by_repeat") or [t["backbone_ms"]] * len(
                t["by_repeat"])
            add, basis = busy, "profile device busy (once)"
        elif use_samp:
            w_src, w_med = times[src]["backbone_ms"], t["backbone_ms"]
            w_reps = t.get("backbone_by_repeat") or [t["backbone_ms"]] * len(
                t["by_repeat"])
            add = (t["draft_ms"] or 0.0)
            basis = ("twin replay wall (no drafter busy in the profile)"
                     if t.get("draft_ms") is not None else None)
        else:
            w_src, w_med, w_reps, add = times[src]["median_ms"], t[
                "median_ms"], t["by_repeat"], 0.0
            basis = ("twin replay wall (no drafter busy in the profile)"
                     if t.get("draft_ms") is not None else None)
        if use_samp:
            add += samp
        fixed = w_src - pf["layers_twin_ms"]
        k = pf["layers_served_ms"] / pf["layers_twin_ms"] if pf[
            "layers_twin_ms"] else 1.0

        def proj(w: float) -> float:
            return fixed + max(w - fixed, 0.0) * k + add

        med = proj(w_med)
        # a repeat's deviation from the median is host-side step jitter (the profiled layers
        # are stable to a few us across steps), so it counts once, unscaled. Scaling it by the
        # layer ratio would multiply host noise by ~13x on the served basis. Per-layer wobble
        # across boots is measured separately (profile_spread).
        out[n] = {
            "median_ms": round(med, 4),
            "by_repeat": [round(med + (w - w_med), 4) for w in w_reps],
            "twin_ms": t["median_ms"],
            "fixed_ms": round(fixed, 4),
            "backbone_ms": t.get("backbone_ms"),
            "draft_ms": t.get("draft_ms"),
            "draft_wall_ms": t.get("draft_ms"),
            "draft_busy_ms": round(busy, 4) if use_busy else None,
            "drafter_basis": basis,
            "sampling_wall_ms": t.get("sampling_wall_ms"),
            "sampling_busy_ms": round(samp, 4) if use_samp else None,
            "layer_scale": round(k, 3),
            "profiles": len(profiles[src]),
            "profile_jobs": len(_by_job(profiles[src])),
            "per_kind_ms": {
                kk: round(v, 4)
                for kk, v in pf["per_kind_ms"].items()
            },
            "profile": "own" if src == n else f"via {src}"
        }
    return out


def _device_step(x: dict) -> dict[str, float] | None:
    """One profiled step's served step from device time only (MODELLED, `served_device`):
    before + after busy + the per-step ops found inside twin layers (once) + sum_k n_k x layer_k
    + drafter busy + sampling busy. None for a profile that lacks the before/after busy (an
    xprof_steps.json written before this basis existed)."""
    if x.get("before_busy_ms") is None or x.get("after_busy_ms") is None:
        return None
    layers = sum(x["served_counts"][k] * x["per_kind_ms"][k]
                 for k in x["served_counts"])
    parts = {
        "before_after": x["before_busy_ms"] + x["after_busy_ms"],
        "per_step_ops": x.get("per_step_ms") or 0.0,
        "layers": layers,
        "drafter": x.get("draft_busy_ms") or 0.0,
        "sampling": x.get("sampling_busy_ms") or 0.0
    }
    parts["total"] = sum(parts.values())
    return parts


def served_device_times(
        cells: dict[str, dict], times: dict[str, dict],
        profiles: dict[str, list]) -> dict[str, dict[str, Any]]:
    """The `served_device` basis (MODELLED): every term of a cell's served step from the profile's
    device time, no host wall anywhere. The `served` basis builds `fixed = backbone wall - layers`
    from the host wall, so it carries the host's per-step jitter (+-0.3 ms on Qwen3.5 at ~10 ms)
    into every cell; on a kernel change that hides a layer-time win of the same size. Here:

        t_i = median over profiles of  (before + after busy + per-step ops + sum_k n_k layer_k
                                        + drafter busy + sampling busy)

    pooled like `served` (median within each job, then over jobs). `by_repeat` holds the
    per-profile values in (job, profile) order, so `evaluate` gives a T_mix per profile index and
    the spread across profiles; `significance` tests two arms on those samples. A cell without a
    profile of its own goes through its class's profiled cell, scaled by the ratio of their
    backbone walls (labelled)."""
    out = {}
    for n, t in times.items():
        if n not in cells:
            continue
        src = n if n in profiles else max(
            (m for m in profiles if m in cells and m in times
             and cells[m]["class"] == cells[n]["class"]),
            key=lambda m: cells[m]["share"],
            default=None)
        if src is None:
            continue
        samples = [(x.get("job", ""), _device_step(x)) for x in profiles[src]]
        samples = [(j, d) for j, d in samples if d is not None]
        if not samples:
            continue
        scale = (t["backbone_ms"] / times[src]["backbone_ms"]
                 if src != n and times[src].get("backbone_ms") else 1.0)
        jobs: dict[str, list[dict]] = {}
        for j, d in samples:
            jobs.setdefault(j, []).append(d)

        def med(key):
            return statistics.median(
                statistics.median(d[key] for d in g) for g in jobs.values())

        parts = {k: round(med(k) * scale, 4) for k in samples[0][1]}
        out[n] = {
            "median_ms":
            parts["total"],
            "by_repeat": [round(d["total"] * scale, 4) for _, d in samples],
            "twin_ms":
            t["median_ms"],
            "backbone_ms":
            t.get("backbone_ms"),
            "parts_ms":
            parts,
            "profiles":
            len(samples),
            "profile_jobs":
            len(jobs),
            "draft_busy_ms":
            parts["drafter"] or None,
            "sampling_busy_ms":
            parts["sampling"] if any(
                x.get("sampling_busy_ms") is not None
                for x in profiles[src]) else None,
            # per job (boot): that boot's median, for the boot-to-boot spread
            "by_job_ms": {
                j: round(statistics.median(d["total"] for d in g) * scale, 4)
                for j, g in jobs.items()
            },
            "profile":
            "own"
            if src == n else f"via {src}, x backbone wall ratio {scale:.3f}"
        }
    return out


def boot_spread(cells: dict[str, dict], dev: dict[str,
                                                  dict]) -> dict[str, Any]:
    """`served_device` T_mix per job (boot) of one arm, over the jobs every measured cell has, and
    their spread: the boot-to-boot noise the within-boot profiles cannot see (dummy routing
    re-draws per boot). None with one boot."""
    have = [n for n in cells if n in dev]
    jobs = set.intersection(*(set(dev[n].get("by_job_ms") or {})
                              for n in have)) if have else set()
    tm = {
        j:
        round(
            sum(cells[n]["steps"] * dev[n]["by_job_ms"][j]
                for n in have) / 1e3, 4)
        for j in sorted(jobs)
    }
    v = list(tm.values())
    med = statistics.median(v) if v else None
    return {
        "t_mix_by_job_s":
        tm,
        "spread_pct":
        round(100 * (max(v) - min(v)) / med, 3) if len(v) > 1 and med else None
    }


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the regularized incomplete beta (Numerical Recipes betacf)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        de = d * c
        h *= de
        if abs(de - 1.0) < 3e-14:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    import math
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(
        b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1.0) / (a + b + 2.0):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lb) * _betacf(b, a, 1.0 - x) / b


def t_two_sided_p(t: float, df: float) -> float:
    """Two-sided p of Student's t with `df` degrees of freedom (no scipy in the harness)."""
    if df <= 0:
        return float("nan")
    return _betai(df / 2.0, 0.5, df / (df + t * t))


def significance(base: list[float],
                 cand: list[float],
                 alpha: float = 0.05,
                 floor_pct: float | None = None) -> dict[str, Any]:
    """Welch's t-test of two arms' per-profile T_mix samples (served_device `t_mix_by_repeat_s`).
    Returns the mean speedup, t, Welch df, two-sided p, a (1 - alpha) interval on the speedup
    (delta-free: the interval of mean(base) - mean(cand), divided through), and `significant`
    = p < alpha and |speedup - 1| beats `floor_pct` (the boot-to-boot noise: an arm's per-boot
    spread or an A/A, when known). Samples within one boot share that boot's dummy-routing mode,
    so the test alone is anticonservative: an A/A of two baseline boots (j-3e69c2dd vs j-36af7327)
    gave 0.9963x at p = 0.014. Without a floor the result says `boot_noise: unmeasured`."""
    import math
    nb, nc = len(base), len(cand)
    if nb < 2 or nc < 2:
        return {
            "test": "welch_t",
            "n_base": nb,
            "n_cand": nc,
            "p_two_sided": None,
            "significant": None,
            "why": "needs >= 2 profiles per arm"
        }
    mb, mc = statistics.mean(base), statistics.mean(cand)
    vb, vc = statistics.variance(base), statistics.variance(cand)
    se2 = vb / nb + vc / nc
    if se2 == 0:
        p, t, df = (0.0 if mb != mc else
                    1.0), float("inf") if mb != mc else 0.0, float(nb + nc - 2)
    else:
        t = (mb - mc) / math.sqrt(se2)
        df = se2 ** 2 / ((vb / nb) ** 2 / (nb - 1) + (vc / nc) ** 2 / (nc - 1)) \
            if vb or vc else float(nb + nc - 2)
        p = t_two_sided_p(t, df)
    # t quantile by bisection on the p function
    lo, hi = 0.0, 100.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    half = hi * math.sqrt(se2)
    ci = [
        round((mc + (mb - mc - half)) / mc, 5),
        round((mc + (mb - mc + half)) / mc, 5)
    ] if mc else None
    return {
        "test":
        "welch_t",
        "n_base":
        nb,
        "n_cand":
        nc,
        "mean_base_s":
        round(mb, 4),
        "mean_cand_s":
        round(mc, 4),
        "speedup_of_means":
        round(mb / mc, 5) if mc else None,
        "t":
        round(t, 3) if math.isfinite(t) else t,
        "df":
        round(df, 2),
        "p_two_sided":
        float(f"{p:.3g}"),
        f"ci{int(100 * (1 - alpha))}_speedup":
        ci,
        "sd_base_pct":
        round(100 * math.sqrt(vb) / mb, 3) if mb else None,
        "sd_cand_pct":
        round(100 * math.sqrt(vc) / mc, 3) if mc else None,
        "boot_noise_floor_pct":
        floor_pct,
        "boot_noise":
        "unmeasured" if floor_pct is None else "floor applied",
        "significant":
        bool(p < alpha
             and (floor_pct is None or abs(mb / mc - 1) * 100 > floor_pct))
    }


def profile_spread(cells: dict[str, dict], times: dict[str, dict],
                   profiles: dict[str, list]) -> dict[str, Any]:
    """T_mix on the served basis with each profiled job (boot) alone: the spread the profile
    sampling puts on T_mix. One job's worth of profiles measures nothing (spread None)."""
    n = max((len(_by_job(v)) for v in profiles.values()), default=0)
    vals = [
        evaluate(cells, served_times(cells, times, profiles,
                                     pick=i))["t_mix_s"] for i in range(n)
    ]
    med = statistics.median(vals) if vals else None
    return {
        "t_mix_by_profile_s": [round(v, 4) for v in vals],
        "profiled_jobs_max":
        n,
        "spread_pct":
        round(100 * (max(vals) - min(vals)) /
              med, 2) if n > 1 and med else None
    }


def evaluate(cells: dict[str, dict], times: dict[str, dict]) -> dict[str, Any]:
    """One arm. `cells`: name -> the plan's cell (class, steps, share); `times`: name ->
    `cell_times` entry. Missing cells are listed, and T_mix covers only the measured ones."""
    have = [n for n in cells if n in times]
    t_mix = sum(cells[n]["steps"] * times[n]["median_ms"] for n in have) / 1e3
    by_class: dict[str, float] = {}
    for n in have:
        c = cells[n]["class"]
        by_class[c] = by_class.get(
            c, 0.0) + cells[n]["steps"] * times[n]["median_ms"] / 1e3
    # T_mix per repeat index: the spread a reader must beat
    reps = min((len(times[n]["by_repeat"]) for n in have), default=0)
    per_rep = [
        sum(cells[n]["steps"] * times[n]["by_repeat"][r] for n in have) / 1e3
        for r in range(reps)
    ]
    spread = (100 * (max(per_rep) - min(per_rep)) / statistics.median(per_rep)
              if len(per_rep) > 1 and statistics.median(per_rep) else None)
    return {
        "t_mix_s": round(t_mix, 4),
        "by_class_s": {
            k: round(v, 4)
            for k, v in by_class.items()
        },
        "t_mix_by_repeat_s": [round(x, 4) for x in per_rep],
        "spread_pct": round(spread, 2) if spread is not None else None,
        "covered_share": round(sum(cells[n]["share"] for n in have), 4),
        "missing": [n for n in cells if n not in times],
        "cells": {
            n: {
                "class":
                cells[n]["class"],
                "steps":
                cells[n]["steps"],
                "share":
                cells[n]["share"],
                "step_ms":
                times[n]["median_ms"],
                "weighted_s":
                round(cells[n]["steps"] * times[n]["median_ms"] / 1e3, 4),
                **{
                    k: times[n][k]
                    for k in ("twin_ms", "fixed_ms", "layer_scale", "profile", "profiles", "per_kind_ms", "backbone_ms", "draft_ms", "draft_wall_ms", "draft_busy_ms", "drafter_basis", "sampling_wall_ms", "sampling_busy_ms", "parts_ms") if k in times[n]
                }
            }
            for n in have
        }
    }


def compare(cells: dict[str, dict], base: dict[str, Any], cand: dict[str, Any],
            busy_s: float | None, idle_s: float | None) -> dict[str, Any]:
    """Candidate vs baseline on the cells both measured.

    speedup          T_mix(base) / T_mix(cand)
    by_class_delta   seconds each class saved (+) or cost (-), and its share of the net change
    guardrails       the heaviest decode cell (typical TPOT) and every mixed cell (TTFT and the
                     TPOT tail: decodes wait through them), each base vs cand
    served_gain      MODELLED: the job's output tok/s if its covered busy time shrank by `speedup`,
                     unmeasured busy time stayed at 1.0x, and idle (client think) time stayed:
                     (busy + idle) / (busy * (covered_share / speedup + 1 - covered_share) + idle)
    significant      |speedup - 1| beats the largest of the arms' repeat spreads and, on the
                     served basis, their profile spreads (`profile_spread`)
    """
    both = [n for n in cells if n in base["cells"] and n in cand["cells"]]
    # an arm with the drafter in its steps against one without compares different work
    drafter = {
        arm:
        sorted({
            c.get("draft_ms") is not None
            for n, c in a["cells"].items() if n in both
        })
        for arm, a in (("base", base), ("cand", cand))
    }
    # likewise the sample stage: an arm that sampled against one that did not
    sampling = {
        arm:
        sorted({
            c.get("sampling_wall_ms") is not None
            or c.get("sampling_busy_ms") is not None
            for n, c in a["cells"].items() if n in both
        })
        for arm, a in (("base", base), ("cand", cand))
    }
    tb = sum(base["cells"][n]["weighted_s"] for n in both)
    tc = sum(cand["cells"][n]["weighted_s"] for n in both)
    sp = tb / tc if tc else None
    delta: dict[str, float] = {}
    for n in both:
        c = cells[n]["class"]
        delta[c] = delta.get(
            c, 0.0
        ) + base["cells"][n]["weighted_s"] - cand["cells"][n]["weighted_s"]
    net = tb - tc
    heavy = max((n for n in both if cells[n]["kind"] == "decode"),
                key=lambda n: cells[n]["share"],
                default=None)
    guard = []
    for n in both:
        if n == heavy or cells[n]["kind"] == "mixed":
            b, c = base["cells"][n]["step_ms"], cand["cells"][n]["step_ms"]
            guard.append({
                "cell": n,
                "role": "typical TPOT" if n == heavy else "TTFT / TPOT tail",
                "base_ms": b,
                "cand_ms": c,
                "ratio": round(b / c, 4) if c else None,
                "regressed": c > b
            })
    noise = max(x
                for x in (base.get("spread_pct"), cand.get("spread_pct"),
                          (base.get("profile_spread") or {}).get("spread_pct"),
                          (cand.get("profile_spread") or {}).get("spread_pct"),
                          0.0) if x is not None)
    cov = min(1.0, max(0.0, sum(cells[n]["share"] for n in both)))
    gain = None
    if sp and cov > 0 and busy_s and idle_s is not None:
        busy_after = busy_s * (cov / sp + (1.0 - cov))
        gain = (busy_s + idle_s) / (busy_after + idle_s)
    return {
        "cells_compared":
        len(both),
        "covered_share":
        round(cov, 4),
        "t_mix_base_s":
        round(tb, 4),
        "t_mix_cand_s":
        round(tc, 4),
        "drafter_in_steps":
        drafter,
        "drafter_mismatch":
        drafter["base"] != drafter["cand"],
        "sampling_in_steps":
        sampling,
        "sampling_mismatch":
        sampling["base"] != sampling["cand"],
        "speedup":
        round(sp, 4) if sp else None,
        "by_class_delta_s": {
            k: round(v, 4)
            for k, v in delta.items()
        },
        "by_class_share_of_change": {
            k: round(v / net, 3)
            for k, v in delta.items()
        } if net else {},
        "guardrails":
        guard,
        "noise_pct":
        noise,
        "significant":
        bool(sp and abs(sp - 1) * 100 > noise),
        "served_output_gain_modelled":
        round(gain, 4) if gain else None,
        "served_gain_rule":
        "(busy + idle) / (busy * (covered_share / speedup + 1 - covered_share) + idle), "
        "the job's measured busy and idle; modelled, not measured"
    }
