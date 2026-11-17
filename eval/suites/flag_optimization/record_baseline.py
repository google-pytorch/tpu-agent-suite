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
"""Record a task's baseline once: assets/baselines/<task_id>.json.

The verifier compares the agent's config against this recording instead of
re-running the baseline in every trial. It is measured exactly as the
verifier measures the candidate (same launcher, steps and warmup), once with
the checkpoint weights and once with random init, each up to `--repeats`
times, keeping the least noisy successful run.

Run it from a machine whose kubectl reaches the task's cluster, e.g.

    python3 eval/suites/flag_optimization/record_baseline.py \\
        flagopt-gemma4-31b-v7x-2x2x2-sharding

The suite stages assets/baselines/ into every task's tests/baselines/.
Re-record whenever the task's image, measurement or baseline config change;
the verifier ignores a recording that does not match the spec.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import tempfile
from pathlib import Path

SUITE_DIR = Path(__file__).resolve().parent
ASSETS = SUITE_DIR / "assets"
sys.dont_write_bytecode = True  # assets/flagopt/ is copied into the image
sys.path.insert(0, str(ASSETS / "flagopt"))
import flagopt  # noqa: E402

KEEP = ("name", "status", "steps", "weights", "exit_code", "summary",
        "metrics", "queued_s", "ran_s", "wall_time_s", "finished_at")


def record(task_id: str, repeats: int, weights_list: list[str],
           runs_dir: Path) -> dict:
    spec = flagopt.load_spec(ASSETS / "flagopt" / "tasks.json", task_id)
    meas = spec["measurement"]
    cluster = flagopt.cluster_settings(spec)
    kube = flagopt.Kube(namespace=cluster["namespace"])
    baseline = flagopt.baseline_candidate(spec)
    out = {
        "task_id": spec["task_id"],
        "recorded_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "image": cluster["image"],
        "cluster": {
            k: cluster[k]
            for k in ("accelerator", "topology", "namespace", "queue")
        },
        "steps": int(meas["steps"]),
        "warmup_steps": int(meas["warmup_steps"]),
        "runs": {},
    }
    for weights in weights_list:
        best = None
        for i in range(repeats):
            res = flagopt.run_candidate(spec,
                                        baseline,
                                        label=f"rec-{weights[:4]}{i}",
                                        steps=out["steps"],
                                        weights=weights,
                                        out_dir=runs_dir,
                                        role="baseline",
                                        kube=kube)
            if not res["ok"]:
                print(
                    f"[record] {weights} run {i} failed: {res.get('status')} "
                    f"{res.get('error_signatures')} {res.get('pod_diagnosis')}"
                )
                continue
            cv = res["summary"]["step_time_cv"]
            if best is None or cv < best["summary"]["step_time_cv"]:
                best = res
            if cv <= meas["max_step_time_cv"]:
                break
        if best is None:
            print(
                f"[record] no successful {weights} baseline run; not recorded")
            continue
        out["runs"][weights] = {k: best[k] for k in KEEP if k in best}
        print(
            f"[record] {weights}: p50 {best['summary']['step_time_p50_s']:.4f}s "
            f"cv {best['summary']['step_time_cv']:.3f} ({best['name']})")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("task_id", help="a task of assets/flagopt/tasks.json")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--weights", default="checkpoint,random")
    ap.add_argument("--runs-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    runs_dir = a.runs_dir or Path(tempfile.mkdtemp(prefix="flagopt-record-"))
    data = record(a.task_id, a.repeats, a.weights.split(","), runs_dir)
    if not data["runs"]:
        print("[record] nothing recorded", file=sys.stderr)
        return 1
    target = ASSETS / "baselines" / f"{a.task_id}.json"
    target.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"[record] wrote {target} (run logs in {runs_dir})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
