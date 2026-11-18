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
"""Step 1 inventory in one pass: steps, step types and census for every trace file.

usage:
    inventory.py [--step-module REGEX] [--top N] [--out steps.csv] <label>=<parquet> ...

  <label>     free text naming the file, for example prefill_only:r10
  <parquet>   events parquet exported by the tpu-diagnostics agent, or its eventsdb JSON

For each TensorCore device in each file (SparseCore and cpu planes are skipped):

  1. Step module. The `XLA Modules` base name (hash suffix removed) holding the
     most device time is the model step. Override with --step-module.
  2. One row per step: file, device, module (the full name, whose hash
     distinguishes batch buckets), t0/t1 ns, wall ms, busy ms (self time of
     the XLA Ops inside the step), op count. Written to --out.
  3. Step types: steps grouped by module, then clustered by op count within 10%. Per
     type: step count, median wall and busy ms, and a census of the median-busy
     step: top-N kernels by self ms, grouped by kernel base name and result
     shape (the `.N` instance suffix is removed), with occurrences per step.

Printed output is capped. Read the step-type tables, name the regimes, then
write windows.csv from the rows in --out. Do not re-run the census by hand.
"""
import argparse
import json
import os
import re
import sys

import duckdb

ap = argparse.ArgumentParser(usage=__doc__)
ap.add_argument(
    "--step-module",
    help="regex over module names; default: the base name with the most time")
ap.add_argument("--top",
                type=int,
                default=15,
                help="kernels listed per step type")
ap.add_argument("--out", default="steps_inventory.csv")
ap.add_argument("files", nargs="+")
a = ap.parse_args()

DUR = "(CAST(end_ns AS BIGINT) - CAST(start_ns AS BIGINT))"
con = duckdb.connect()
rows = []

for spec in a.files:
    if "=" not in spec:
        sys.exit(f"expected <label>=<parquet>, got {spec}")
    label, src = spec.split("=", 1)
    if src.endswith(".json"):
        src = json.load(open(src))["path"]
    if not os.path.exists(src):
        sys.exit(f"no such parquet: {src}")
    con.execute(
        f"CREATE OR REPLACE VIEW E AS SELECT * FROM read_parquet('{src}')")
    devs = [
        d for (d, ) in con.execute(
            "SELECT DISTINCT device FROM E WHERE device LIKE 'TPU:%' AND device NOT LIKE '%SparseCore%' ORDER BY 1"
        ).fetchall()
    ]
    print(f"\n#### {label}  devices={devs}")
    for dev in devs:
        mods = con.execute(f"""
            SELECT kernel_name, count(*) n, sum({DUR})/1e6 ms FROM E
            WHERE device='{dev}' AND category='XLA Modules' GROUP BY 1"""
                           ).fetchall()
        if not mods:
            print(f"  {dev}: no XLA Modules events")
            continue
        if a.step_module:
            keep = [m for m, _, _ in mods if re.search(a.step_module, m)]
        else:

            def base(m):
                return re.sub(r"\(\d+\)$", "", m)

            tot = {}
            for m, _, ms in mods:
                tot[base(m)] = tot.get(base(m), 0) + ms
            best = max(tot, key=tot.get)
            keep = [m for m, _, _ in mods if base(m) == best]
        inlist = ",".join("'" + m.replace("'", "''") + "'" for m in keep)
        steps = con.execute(f"""
            WITH m AS (SELECT kernel_name module, start_ns s, end_ns e, {DUR}/1e6 wall
                       FROM E WHERE device='{dev}' AND category='XLA Modules' AND kernel_name IN ({inlist}))
            SELECT m.module, m.s, m.e, m.wall, count(x.start_ns) nops,
                   coalesce(sum(x.self_time_ns), 0)/1e6 busy
            FROM m LEFT JOIN E x ON x.device='{dev}' AND x.category='XLA Ops'
                 AND x.start_ns >= m.s AND x.end_ns <= m.e
            GROUP BY 1,2,3,4 ORDER BY m.s""").fetchall()
        # Step types: per module, sort steps by op count and start a new type when the
        # count exceeds the first step of the current type by more than 10%.
        types = {}
        for i, (mod, s, e, wall, nops, busy) in enumerate(steps):
            rows.append((label, dev, mod, i, s, e, round(wall,
                                                         3), round(busy,
                                                                   3), nops))
        for mod in sorted({st[0] for st in steps}):
            order = sorted((i for i, st in enumerate(steps) if st[0] == mod),
                           key=lambda j: steps[j][4])
            first = None
            for i in order:
                if first is None or steps[i][4] > 1.1 * max(
                        steps[first][4], 1):
                    first = i
                types.setdefault((mod, first), []).append(i)
        for idx in types.values():
            idx.sort()
        print(f"  {dev}: {len(steps)} steps, {len(types)} step types")
        for (mod, _), idx in sorted(types.items(), key=lambda kv: -len(kv[1])):
            ws = sorted(steps[i][3] for i in idx)
            bs = sorted(steps[i][5] for i in idx)
            ns = sorted(steps[i][4] for i in idx)

            def med(v):
                return v[len(v) // 2]

            print(
                f"    type module={mod[-60:]} steps={len(idx)} step_idx={idx[:8]}"
                f" wall_ms={med(ws):.2f} busy_ms={med(bs):.2f} nops={med(ns)}")
            # Census on the median-busy step of this type only: the tables are for naming regimes.
            i = sorted(idx, key=lambda j: steps[j][5])[len(idx) // 2]
            _, s, e, *_ = steps[i]
            cen = con.execute(f"""
                SELECT left(regexp_replace(kernel_name, '^%([^ ]*?)\\.[0-9]+ = ', '%\\1 = '), 100) k,
                       count(*) n, sum(self_time_ns)/1e6 ms
                FROM E WHERE device='{dev}' AND category='XLA Ops' AND start_ns >= {s} AND end_ns <= {e}
                GROUP BY 1 ORDER BY ms DESC LIMIT {a.top}""").fetchall()
            for k, n, ms in cen:
                print(f"      {ms:9.3f} ms  n={n:<6d} {k}")

with open(a.out, "w") as fh:
    fh.write("label,device,module,step_idx,t0_ns,t1_ns,wall_ms,busy_ms,nops\n")
    for r in rows:
        fh.write(",".join('"' + str(x).replace('"', "'") +
                          '"' if isinstance(x, str) else str(x)
                          for x in r) + "\n")
print(f"\n{len(rows)} steps written to {a.out}")
