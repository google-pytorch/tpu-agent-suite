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
"""A finished forward_run job -> evidence.json: the facts a report is written from.

Data only: numbers, shapes, ids and cross-checks. No prose, no layout. The report itself is
written by the agent (SKILL.md, "REPORT.md"), which decides what to explain for each run.

Written by `cli.py collect` next to the job's result.json. Contents:
  plan / result       paths of the plan the job ran and its result.json
  layers              twin layer kinds in order, served count per kind, served layer ids per
                      kind (from the model's own model.py), MTP layer count
  sharding            served and twin labels, twin world, context scale, twin hf_overrides
  projection          per-rank served-vs-twin table and deviations (from the twin config)
  layer_census_check  per profiled step and layer: does the layer contain what its kind says
                      (indexer ops in attention iff *_idx; expert ops in the FFN iff moe*)? This
                      is what tests the boundary rule: a wrong rule moves ops between layers
  xprof_categories    the HLO op profile by category, busy time only, from the
                      `hlo_categories.json` tpu-diagnostics wrote next to result.json
  offload             SparseCore offload threshold the pod ran with; every profiled collective
                      with its size and which side of the threshold it falls on, twin and served
  pb                  rank-0 .xplane.pb path
  patches             the vllm_torchtpu patch stack the job applied (forward-cmd --patch), each
                      patch's sha256 and applier exit code, and the outcome
"""

from __future__ import annotations

import glob
import json
import os
import re
from typing import Any

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
_BYTES = {
    "bf16": 2,
    "f16": 2,
    "f32": 4,
    "s32": 4,
    "u32": 4,
    "s8": 1,
    "u8": 1,
    "f8e4m3fn": 1
}


def _find_one(job_dir: str, pattern: str) -> str | None:
    hits = sorted(
        glob.glob(os.path.join(job_dir, "outputs", "**", pattern),
                  recursive=True))
    return hits[-1] if hits else None


WHAT = "evidence for REPORT.md (data only; the agent writes the report)"
#: the line a forward-cmd recipe prints after each `apply_vllm_patch.sh`
_PATCH_LINE = re.compile(r"WMB_PATCH name=(\S+) sha256=(\w+) exit=(-?\d+)\s*$")


def patches(job_dir: str) -> dict:
    """The patch stack from the marker lines in the job's fetched log (`<job>.log`). Applier
    exit 4 is "patch does not apply" (base mismatch); 2 and 3 are infrastructure failures."""
    stack = []
    for path in sorted(glob.glob(os.path.join(job_dir, "*.log"))):
        for line in open(path, errors="replace"):
            if "WMB_PATCH" not in line:
                continue
            m = _PATCH_LINE.search(line)
            if m:
                stack.append({
                    "name": m.group(1),
                    "sha256": m.group(2),
                    "exit": int(m.group(3))
                })
    rc = next((p["exit"] for p in stack if p["exit"]), 0)
    outcome = None if not stack else {
        0: "applied",
        4: "patch does not apply"
    }.get(rc, "infrastructure failure")
    return {"stack": stack, "exit": rc, "outcome": outcome}


def fetch_rank0(job_dir: str) -> str | None:
    """The rank-0 .xplane.pb of a profiled job, as fetched (rank 0 only: the others are not
    read)."""
    return _find_one(job_dir, "wmb/forward/xprof/**/rank0_*.xplane.pb")


def xprof_categories(res_dir: str) -> list[dict] | None:
    """The HLO op profile by category from `hlo_categories.json` (written by tpu-diagnostics:
    a list of {category, total_self_time_ms, op_count, top_op}), busy time only."""
    path = os.path.join(res_dir, "hlo_categories.json")
    if not os.path.isfile(path):
        return None
    try:
        raw = json.load(open(path))
        cats = raw.get("category_summary", raw) if isinstance(raw,
                                                              dict) else raw
        busy = sum(c["total_self_time_ms"] for c in cats
                   if c["category"] != "IDLE")
        return [{
            "category": c["category"],
            "ms": round(c["total_self_time_ms"], 3),
            "pct_of_busy": round(100 * c["total_self_time_ms"] / busy, 2),
            "ops": c.get("op_count"),
            "top_op": c.get("top_op")
        } for c in cats if c["category"] != "IDLE"]
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None


def mib(shape: str) -> float | None:
    m = re.match(r"([a-z0-9]+)\[([\d,]*)\]", shape or "")
    if not m or m.group(1) not in _BYTES:
        return None
    n = 1
    for x in filter(None, m.group(2).split(",")):
        n *= int(x)
    return round(n * _BYTES[m.group(1)] / 2**20, 3)


def offload(eff_env: list[dict], xsteps: list[dict], twin_world: int,
            real_world: int) -> dict:
    """SparseCore offload threshold the pod ran with, and each profiled all-gather / all-reduce
    against it. A gathered output grows with the group, so the served size is the twin size
    x real/twin for all-gathers; all-reduce outputs do not change with the group."""
    v = next((e["value"] for e in eff_env if e["name"] == "LIBTPU_INIT_ARGS"),
             "")
    ar = re.search(r"sparse_core_all_reduce_offload_min_size_in_bytes=(\d+)",
                   v)
    ag = re.search(r"sparse_core_all_gather_offload_min_size_in_bytes=(\d+)",
                   v)
    thr = {
        "all_reduce_mib": int(ar.group(1)) / 2**20 if ar else None,
        "all_gather_mib": int(ag.group(1)) / 2**20 if ag else None
    }
    rows = []
    for sp in xsteps or []:
        segs = [("before", sp.get("before"))] + [
            (f"L{i}", layer) for i, layer in enumerate(sp.get("layers", []))
        ]
        for nm, seg in segs:
            for c in (seg or {}).get("collectives", []):
                size = mib(c["shape"])
                gather = "gather" in c["op"]
                kind = "all_gather_mib" if gather else "all_reduce_mib" if "all-reduce" in c[
                    "op"] else None
                t = thr.get(kind) if kind else None
                served = (size * real_world /
                          twin_world if twin_world else None) if (
                              size and gather) else size
                rows.append({
                    "step": sp["label"],
                    "segment": nm,
                    "op": c["op"],
                    "shape": c["shape"],
                    "twin_mib": size,
                    "served_mib": served,
                    "threshold_mib": t,
                    "twin_on_sc": bool(t and size and size >= t),
                    "served_on_sc": bool(t and served and served >= t)
                })
    return {
        "threshold": thr,
        "collectives": rows,
        "flips": [r for r in rows if r["twin_on_sc"] != r["served_on_sc"]]
    }


def _deg(label: str, k: str) -> int:
    m = re.search(rf"{k}(\d+)", label or "")
    return int(m.group(1)) if m else 1


def projection(plan: dict) -> tuple[list | None, list | None]:
    if plan.get("projection"):
        return plan["projection"], plan.get("deviations")
    gen = (plan.get("_header") or {}).get("generated_by", "")
    m = re.match(r"points (\S+\.json)(.*)$", gen)
    if not m or "--override" in m.group(2):
        return None, None
    path = os.path.join(SCRIPTS, m.group(1))
    if not os.path.isfile(path):
        return None, None
    cfg = json.load(open(path))
    return cfg.get("projection"), cfg.get("deviations")


def layers(plan: dict, plan_path: str) -> dict:
    """Twin layer kinds, served counts, and served layer ids per kind (model.py's own rule)."""
    out = {
        "twin": plan["layers"]["twin"],
        "served_counts": plan["layers"]["real"],
        "served_total": sum(plan["layers"]["real"].values())
    }
    try:
        import sys
        from dataclasses import asdict
        sys.path.insert(0, SCRIPTS)
        from core import derive as derive_mod
        model_dir = plan.get("model_dir") or plan_path.split(
            "/.work/", 1)[1].split("/", 1)[0]
        f = derive_mod.load_model(model_dir).facts(
            plan["checkpoint_config"],
            str(plan["twin"]["server_args"].get("kv-cache-dtype", "auto")))
        ids: dict[str, list[int]] = {}
        for i, k in enumerate(f.kinds):
            ids.setdefault(k, []).append(i)
        out["served_ids"] = ids
        out["mtp_layers"] = f.mtp_layers
        # every canonical fact the model filled in (non-empty), for the report's setup section
        out["served_facts"] = {
            k: v
            for k, v in asdict(f).items()
            if k != "kinds" and v not in (0, (), None, "")
        }
    except Exception as e:
        out["served_ids_error"] = repr(e)[:200]
    return out


def build(job_dir: str, plan_path: str) -> dict[str, Any]:
    res = _find_one(job_dir, "wmb/forward/result.json")
    r = json.load(open(res))
    ran = os.path.join(os.path.dirname(res),
                       "plan.steps.json")  # saved by forward_run
    # the trace rule is how to READ the profile, not part of the run: take it from the plan
    # `collect` was given, as xprof_steps.py does, so a corrected rule re-reads an old job
    given_trace = json.load(
        open(plan_path)).get("trace") if os.path.isfile(plan_path) else None
    plan_path = ran if os.path.isfile(ran) else plan_path
    plan = json.load(open(plan_path))
    if given_trace:
        plan["trace"] = given_trace
    xs_path = os.path.join(os.path.dirname(res), "xprof_steps.json")
    xs = json.load(open(xs_path)) if os.path.isfile(xs_path) else {}
    steps = xs.get("steps") or []
    shard = plan.get("sharding", {})
    rs = shard.get("real", "")
    real_world = _deg(rs, "tp") * _deg(rs, "dp") * _deg(rs, "pp")
    twin_world = shard.get("twin_world")
    pb = fetch_rank0(job_dir) if r.get("profile_dir") else None
    proj, dev = projection(plan)
    lay = layers(plan, plan_path)
    return {
        "_what":
        WHAT,
        "job_id":
        os.path.basename(job_dir.rstrip("/")),
        "result":
        res,
        "plan":
        plan_path,
        "xprof_steps":
        xs_path if xs else None,
        "pb":
        pb,
        "sharding": {
            **shard, "real_world": real_world,
            "context_scale": plan.get("context_scale"),
            "twin_hf_overrides": {
                k: v
                for k, v in plan["twin"]["hf_overrides"].items()
                if k != "quantization_config"
            },
            "twin_server_args": plan["twin"]["server_args"]
        },
        "layers":
        lay,
        "projection":
        proj,
        "deviations":
        dev,
        "plan_is_the_one_run":
        plan_path.endswith("plan.steps.json"),
        "image_source":
        r.get("image_source"),
        "layer_census_check": [{
            "step": sp["label"],
            "layers": sp.get("census_check")
        } for sp in steps],
        "xprof_categories":
        xprof_categories(os.path.dirname(res)),
        "offload":
        offload(r.get("effective_env") or [], steps, twin_world, real_world),
        "patches":
        patches(job_dir),
    }
