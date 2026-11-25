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
"""Minimal unit tests for `framework.hc.metrics`."""

from __future__ import annotations

import statistics
import unittest

from framework import jsonio
from framework.hc import constants, db, metrics, paths, telemetry
from framework.tests.hc import _helpers


class TestMetrics(_helpers.HcCliTestCase):
    """Tests canonical `summary.json` parsing, validation, and trial pooling."""

    def test_pooling_and_db_persistence(self) -> None:
        self.assertEqual(
            self._run_hc("campaign", "open", str(self.camp_json),
                         str(self.proto_json), "--code-dir",
                         str(self.code_dir))[0],
            0,
        )
        with db.open_db() as conn:
            with db.transaction(conn):
                conn.execute(
                    "INSERT INTO Jobs (job_id, candidate, protocol, trial, kind, status) "
                    "VALUES ('j1', 1, 1, 't1', 'E2E_Benchmark', 'Succeeded'), "
                    "('j2', 1, 1, 't1', 'E2E_Benchmark', 'Succeeded')")
            proto = conn.execute(
                "SELECT * FROM Protocols WHERE id = 1").fetchone()
            cells = conn.execute(
                "SELECT * FROM Protocol_Cells WHERE protocol_id = 1").fetchall(
                )

            s1, s2 = self.base / "s1.json", self.base / "s2.json"
            jsonio.write_json(
                s1, {
                    "status":
                    "ok",
                    "environment": {
                        "jax_version": "0.4.35"
                    },
                    "cells": [{
                        "cell_index": 0,
                        "status": "ok",
                        "replicates": [{
                            "output_tokens_per_s": 950.0
                        }],
                    }],
                })
            jsonio.write_json(
                s2, {
                    "status":
                    "ok",
                    "environment": {
                        "jax_version": "0.4.35"
                    },
                    "cells": [
                        {
                            "isl": 1024,
                            "osl": 1024,
                            "concurrency": 16,
                            "status": "ok",
                            "replicates": [{
                                "output_tokens_per_s": 1050.0
                            }],
                        },
                        {
                            "isl": 4096,
                            "osl": 1024,
                            "concurrency": 64,
                            "status": "skipped",
                            "replicates": [],
                        },
                    ],
                })
            p1 = metrics.parse_benchmark_summary(job_id="j1",
                                                 summary_path=s1,
                                                 protocol_row=proto,
                                                 protocol_cells=cells)
            p2 = metrics.parse_benchmark_summary(job_id="j2",
                                                 summary_path=s2,
                                                 protocol_row=proto,
                                                 protocol_cells=cells)

            p2_mismatch = metrics.ParsedBenchmarkJob(
                "j2", {
                    **p2.environment, "jax_version": "0.4.36"
                }, p2.artifacts, p2.cell_metrics)
            with self.assertRaises(constants.ValidationError):
                metrics.pool_parsed_jobs([p1, p2_mismatch], proto, cells)

            telem = telemetry.TelemetryBuffer(run_id=self.run_id,
                                              campaign_id=1)
            mid = metrics.record_pooled_measurement(
                conn,
                campaign_id=1,
                round_id=None,
                trial="t1",
                candidate_id=1,
                protocol_row=proto,
                protocol_cells=cells,
                parsed_jobs=[p1, p2],
                trigger_job_id="j2",
                telem=telem,
            )
            telem.flush()

            row = conn.execute(
                "SELECT value, n, stdev FROM Measurement_Metrics WHERE measurement_id = ?",
                (mid, ),
            ).fetchone()
            self.assertEqual(row["n"], 2)
            self.assertAlmostEqual(row["value"], 1000.0)
            self.assertAlmostEqual(row["stdev"],
                                   statistics.stdev([950.0, 1050.0]))

        events = jsonio.read_jsonl(paths.get_campaign_events_log(1))
        self.assertEqual(events[-1]["event_name"], "measurement.recorded")

    def test_validation_and_accuracy_summary(self) -> None:
        obs = [
            metrics.MetricObservation.from_samples([10.0, 30.0]),
            metrics.MetricObservation.from_samples([20.0]),
        ]
        self.assertAlmostEqual(
            metrics.pool_observations(obs,
                                      constants.MetricStatistic.MEDIAN).value,
            20.0,
        )
        self.assertEqual(
            metrics.normalize_environment(
                exec_image="img@sha256:abcd")["image_digest"],
            "sha256:abcd",
        )

        proto = {
            "workload_kind": "inference",
            "primary_metric": "tok_s",
            "exec_image": "img"
        }
        cells = [{
            "id": 1,
            "cell_index": 0,
            "topology": "2x4",
            "prefill": 1024,
            "decode": 1024,
            "concurrency": 16
        }]
        bad = self.base / "bad.json"
        for doc in (
                '{"status":"ok","cells":[{"cell_index":5,"replicates":[{"tok_s":1.0}]}]}',
                '{"status":"ok","cells":[{"cell_index":0,"replicates":[{"tok_s":1.0}]},{"cell_index":0,"replicates":[{"tok_s":2.0}]}]}',
                '{"status":"ok","cells":[{"cell_index":0,"replicates":[{"tok_s":Infinity}]}]}',
        ):
            bad.write_text(doc, encoding="utf-8")
            with self.assertRaises(constants.ValidationError):
                metrics.parse_benchmark_summary(job_id="jb",
                                                summary_path=bad,
                                                protocol_row=proto,
                                                protocol_cells=cells)

        acc = self.base / "acc.json"
        jsonio.write_json(acc, {"status": "ok", "scores": {"em": 0.82}})
        self.assertAlmostEqual(
            metrics.parse_accuracy_summary(job_id="ja",
                                           summary_path=acc,
                                           metric_name="em"),
            0.82,
        )
        jsonio.write_json(acc, {"status": "failed", "scores": {"em": 0.82}})
        with self.assertRaises(constants.ValidationError):
            metrics.parse_accuracy_summary(job_id="ja",
                                           summary_path=acc,
                                           metric_name="em")


if __name__ == "__main__":
    unittest.main()
