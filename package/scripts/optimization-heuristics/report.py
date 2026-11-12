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
"""Per-point results of measured candidates against the baseline.

    python3 report.py [--candidates <work>/candidates.json] \\
        --baseline <baseline bench>/summary.json \\
        --measured <cand_id>=<candidate bench>/summary.json [...] \\
        --out <work>/report.json

Each summary.json is serving_benchmark.py's. At every (isl, osl,
concurrency) cell both measured, for output tokens/s (higher is better) and
median TTFT and ITL (lower is better), it gives the baseline's and the
candidate's median over replicates, the change, the run-to-run noise and a
verdict from `evaluate_arm.evaluate_arm`: `win` or `loss` beyond the noise,
else `tie`. The noise is the larger replicate spread ((max - min) / median)
of the two runs at that cell, never below 1%; with one replicate on either
side it is not measured, 3% is used, and the cell says so. Failed requests
are listed per cell.

Each candidate also gets a summary per traffic regime (one ISL/OSL pair at
all its concurrencies): where throughput won and lost, its mean change, and
where TTFT or ITL lost, so a caller can keep a different configuration for
each kind of traffic. `peak_point` is the concurrency where the baseline's
throughput peaked in that regime, and `at_peak` the candidate's numbers and
verdicts there: the point a caller with no stated target judges at.

It reports; it decides nothing. Whoever ran the round keeps or drops each
candidate from these numbers.
"""

import argparse
import json
import os
import statistics
import sys
from typing import Any

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(
    0, os.path.join(os.path.dirname(_HERE), "tpu-performance-optimization"))

# pylint: disable=g-import-not-at-top,wrong-import-position
import evaluate_arm  # noqa: E402

# Reported metric -> direction.
METRICS = {
    "output_tokens_per_s": "maximize",
    "ttft_p50_ms": "minimize",
    "itl_p50_ms": "minimize",
}
MIN_NOISE_PCT, ASSUMED_NOISE_PCT = 1.0, 3.0

Cell = tuple[int, int, int]


def label(p: Cell) -> str:
    return f"{p[0]}:{p[1]}:{p[2]}"


def cells(summary: dict[str, Any]) -> dict[Cell, dict[str, Any]]:
    """Per (isl, osl, concurrency): each metric's replicates, and failures."""
    out = {}
    for cell in summary.get("cells") or []:
        replicates = cell.get("replicates") or []
        out[(int(cell["isl"]), int(cell["osl"]), int(cell["concurrency"]))] = {
            "values": {
                metric: [
                    float(rep[metric]) for rep in replicates
                    if rep.get(metric) is not None
                ]
                for metric in METRICS
            },
            "failed":
            int(sum(float(rep.get("failed") or 0) for rep in replicates)),
        }
    return out


def _spread_pct(values: list[float]) -> float | None:
    if len(values) < 2 or not statistics.median(values):
        return None
    return 100.0 * (max(values) - min(values)) / statistics.median(values)


def compare(cand_id: str, base: dict, cand: dict) -> list[dict[str, Any]]:
    """One row per cell both runs measured."""
    rows = []
    for point in sorted(set(base) & set(cand)):
        row: dict[str, Any] = {
            "point": label(point),
            "failed_baseline": base[point]["failed"],
            "failed_candidate": cand[point]["failed"],
        }
        for metric, direction in METRICS.items():
            base_values = base[point]["values"][metric]
            cand_values = cand[point]["values"][metric]
            if not base_values or not cand_values:
                continue
            spreads = [
                spread for spread in (_spread_pct(base_values),
                                      _spread_pct(cand_values))
                if spread is not None
            ]
            measured = len(spreads) == 2
            noise = (max(max(spreads), MIN_NOISE_PCT)
                     if measured else ASSUMED_NOISE_PCT)
            verdict = evaluate_arm.evaluate_arm(
                f"{cand_id}@{label(point)}:{metric}",
                "optimization-heuristics", cand_values, {
                    "primary_metric": {
                        "name": metric,
                        "direction": direction
                    },
                    "noise_floor_pct": noise,
                    "replicate_count": 1,
                }, {"metric": {
                    "value": statistics.median(base_values)
                }})
            row[metric] = {
                "baseline": statistics.median(base_values),
                "candidate": statistics.median(cand_values),
                "delta_pct": (verdict.get("metric") or {}).get("delta_pct"),
                "noise_pct": round(noise, 2),
                "noise_measured": measured,
                "verdict": verdict.get("verdict"),
            }
        rows.append(row)
    return rows


def regimes(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per ISL/OSL regime: throughput wins, losses, mean change; latency losses."""
    out: dict[str, dict[str, Any]] = {}
    deltas: dict[str, list[float]] = {}
    for row in rows:
        isl, osl, _ = row["point"].split(":")
        key = f"{isl}:{osl}"
        group = out.setdefault(
            key, {
                "points": [],
                "throughput_wins_at": [],
                "throughput_losses_at": [],
                "ttft_losses_at": [],
                "itl_losses_at": [],
                "failed_requests": 0,
            })
        group["points"].append(row["point"])
        group["failed_requests"] += row["failed_candidate"]
        throughput = row.get("output_tokens_per_s") or {}
        if throughput.get("verdict") == "win":
            group["throughput_wins_at"].append(row["point"])
        elif throughput.get("verdict") == "loss":
            group["throughput_losses_at"].append(row["point"])
        if throughput.get("delta_pct") is not None:
            deltas.setdefault(key, []).append(throughput["delta_pct"])
        for metric, field in (("ttft_p50_ms", "ttft_losses_at"),
                              ("itl_p50_ms", "itl_losses_at")):
            if (row.get(metric) or {}).get("verdict") == "loss":
                group[field].append(row["point"])
    for key, group in out.items():
        d = deltas.get(key)
        group["mean_throughput_delta_pct"] = (round(statistics.mean(d), 2)
                                              if d else None)
        mine = [row for row in rows if row["point"] in group["points"]]
        peak = max(mine,
                   key=lambda row:
                   (row.get("output_tokens_per_s") or {}).get("baseline", 0.0),
                   default=None)
        if peak is not None:
            group["peak_point"] = peak["point"]
            group["at_peak"] = {
                metric: {
                    k: (peak.get(metric) or {}).get(k)
                    for k in ("baseline", "candidate", "delta_pct", "verdict")
                }
                for metric in METRICS
            }
            group["at_peak"]["failed_requests"] = peak["failed_candidate"]
    return out


def report(candidates: dict[str, Any], baseline: dict[str, Any],
           measured: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base = cells(baseline)
    out = []
    for c in candidates.get("candidates") or []:
        cid = str(c["cand_id"])
        entry: dict[str, Any] = {
            "cand_id": cid,
            "kind": c.get("kind"),
            "status": c.get("status")
        }
        if cid not in measured:
            entry["measured"] = False
            out.append(entry)
            continue
        cand = cells(measured[cid])
        rows = compare(cid, base, cand)
        tput = "output_tokens_per_s"
        entry.update({
            "measured":
            True,
            "points":
            rows,
            "missing_points":
            [label(p) for p in sorted(set(base) - set(cand))],
            "throughput_wins_at": [
                r["point"] for r in rows
                if (r.get(tput) or {}).get("verdict") == "win"
            ],
            "throughput_losses_at": [
                r["point"] for r in rows
                if (r.get(tput) or {}).get("verdict") == "loss"
            ],
            "failed_requests":
            sum(r["failed_candidate"] for r in rows),
            "regimes":
            regimes(rows),
        })
        out.append(entry)
    return {"metrics": METRICS, "candidates": out}


def load_summaries(paths: list[str]) -> dict[str, Any]:
    """One summary from several serving_benchmark summary.json files.

    A benchmark run one point per job leaves one summary per point; their
    cells are merged, the last file winning when two measure the same point.
    """
    merged: dict[Cell, dict[str, Any]] = {}
    for path in paths:
        with open(path, encoding="utf-8") as f:
            summary = json.load(f)
        for cell in summary.get("cells") or []:
            key = (int(cell["isl"]), int(cell["osl"]),
                   int(cell["concurrency"]))
            merged[key] = cell
    return {"cells": [merged[k] for k in sorted(merged)]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--candidates",
                        default=None,
                        help="The candidates.json the measured candidates"
                        " came from (propose.py's output); without it, each"
                        " --measured id is a candidate")
    parser.add_argument("--baseline",
                        action="append",
                        required=True,
                        help="The baseline's summary.json; repeat (or join"
                        " with commas) when each point has its own")
    parser.add_argument("--measured",
                        action="append",
                        default=[],
                        help="CAND_ID=summary.json[,summary.json...]; repeat"
                        " per candidate")
    parser.add_argument("--out",
                        required=True,
                        help="Where to write the per-point report.json")
    args = parser.parse_args(argv)
    if args.candidates:
        with open(args.candidates, encoding="utf-8") as f:
            candidates = json.load(f)
    else:
        candidates = {
            "candidates": [{
                "cand_id": m.partition("=")[0]
            } for m in args.measured]
        }
    baseline = load_summaries(
        [p for arg in args.baseline for p in arg.split(",") if p])
    if not baseline["cells"]:
        parser.error("--baseline has no cells: pass serving_benchmark"
                     " summary.json files")
    measured = {}
    for m in args.measured:
        cid, _, paths = m.partition("=")
        if not paths:
            parser.error(f"--measured {m!r} must be CAND_ID=summary.json")
        measured[cid] = load_summaries([p for p in paths.split(",") if p])
        common = set(cells(baseline)) & set(cells(measured[cid]))
        if not common:
            parser.error(
                f"{cid}: no (isl, osl, concurrency) point both the baseline"
                " and the candidate measured; pass each one's"
                " serving_benchmark summary.json files")
    out = report(candidates, baseline, measured)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    for c in out["candidates"]:
        if not c.get("measured"):
            print(f"{c['cand_id']}: not measured")
            continue
        print(f"{c['cand_id']}: throughput wins at "
              f"{', '.join(c['throughput_wins_at']) or 'none'}; losses at "
              f"{', '.join(c['throughput_losses_at']) or 'none'}; "
              f"{c['failed_requests']} failed requests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
