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
"""Run a SQL query against an xprof Events DB parquet using DuckDB.

Use this instead of XProf's own events query command. That command holds a
per-user SQLite result cache in `tempfile.gettempdir()` for the whole
computation, so two concurrent invocations fail with `database is locked`.

usage:
    q.py <parquet-path | eventsdb.json> "<SQL over the view Events>"

The parquet is produced once per trace file by the tpu-diagnostics agent, which
exports XProf's events database on request. This script never runs XProf.

Column notes:
  - `start_ns` and `end_ns` are UINT64. Cast to BIGINT before subtracting.
  - `input_tensors` and `output_tensors` are VARCHAR[]. Use
    list_aggregate(input_tensors, 'string_agg', ',') before LIKE.
"""
import json
import os
import sys

import duckdb
import pandas as pd

if len(sys.argv) != 3:
    sys.exit(__doc__)

src, sql = sys.argv[1], sys.argv[2]
if src.endswith(".json"):
    with open(src) as fh:
        src = json.load(fh)["path"]
if not os.path.exists(src):
    sys.exit(f"no such parquet: {src}")

con = duckdb.connect()
con.execute(f"CREATE VIEW Events AS SELECT * FROM read_parquet('{src}')")
df = con.execute(sql).fetchdf()
pd.set_option("display.width", 250, "display.max_columns", 60,
              "display.max_colwidth", 70, "display.max_rows", 300)
print(df.to_string(index=False))
