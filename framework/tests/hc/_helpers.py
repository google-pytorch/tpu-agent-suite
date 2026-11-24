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
"""Shared CLI test fixtures and config builders for `framework.tests.hc`."""

from __future__ import annotations

import contextlib
import io
import os
import pathlib
import tempfile
import unittest
from unittest import mock

from framework import jsonio
from framework import paths as base_paths
from framework import runtime
from framework.hc import db

Path = pathlib.Path


def make_protocol_cfg(*, is_micro: bool = False) -> dict[str, object]:
    gate = (None if is_micro else dict(
        metric="exact_match,custom-extract",
        comparator="gte",
        threshold=-0.015,
        is_relative=True,
        relative_to=None,
    ))
    return dict(
        workload_kind="inference",
        primary_metric="output_tokens_per_s",
        compute_profile="tpu-v6e-8",
        primary_metric_unit="tok/s",
        primary_metric_direction="maximum",
        primary_metric_statistic="mean",
        measurement_cells=[
            dict(topology="2x4",
                 weight=1.0,
                 prefill=1024,
                 decode=1024,
                 concurrency=16)
        ],
        min_replicates=2,
        min_win_percentage=2.0,
        max_cell_regression_percentage=1.5,
        noise_floor_percentage=0.5,
        max_rounds=10,
        max_barren_rounds=3,
        target_improvement_percent=15.0,
        accuracy_gate=gate,
        execution_spec=dict(
            image="us-docker.pkg.dev/proj/vllm:latest",
            microbenchmark_target="micro_plan.steps.json"
            if is_micro else None,
            base_server_args="--tensor-parallel-size 8",
            base_client_args="--num-prompts 128",
            base_eval_args=None if is_micro else "--num_fewshot 5",
        ),
    )


class HcCliTestCase(unittest.TestCase):
    """Shared isolated store/workspace fixture for `runtime.py hc` CLI tests."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.base = Path(self._tmp.name)
        self.home = self.base / "home"
        self.workspace = self.base / "workspace"
        self.code_dir = self.base / "code"
        for d in (self.home, self.workspace, self.code_dir):
            d.mkdir()
        (self.code_dir / "model.py").write_text("TILE_M = 128\nTILE_N = 128\n",
                                                encoding="utf-8")
        (self.workspace / "micro_plan.steps.json").write_text(
            '{"points": []}\n', encoding="utf-8")

        env_patch = mock.patch.dict(os.environ,
                                    {"TPU_AGENT_SUITE_HOME": str(self.home)})
        env_patch.start()
        self.addCleanup(env_patch.stop)

        self.run_id = "20261110T170000Z-0123456789ab"
        base_paths.get_run_root(self.run_id).mkdir(parents=True, exist_ok=True)
        base_paths.record_active_run(self.run_id, self.workspace)

        self.camp_json = self.base / "campaign.json"
        self.proto_json = self.base / "protocol.json"
        self.micro_json = self.base / "micro_protocol.json"
        jsonio.write_json(
            self.camp_json,
            dict(name="llama3-70b-v6e",
                 owner="researcher",
                 target_model="meta-llama/Llama-3.3-70B-Instruct",
                 engine="vllm",
                 accelerator="v6e"),
        )
        jsonio.write_json(self.proto_json, make_protocol_cfg(is_micro=False))
        jsonio.write_json(self.micro_json, make_protocol_cfg(is_micro=True))

    def _run_hc(self, *args: str) -> tuple[int, str]:
        out_buf, err_buf = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(
                err_buf):
            rc = runtime.main(
                ["hc", "--workspace",
                 str(self.workspace), *args])
        return rc, out_buf.getvalue().strip()

    def _anchor_campaign(self, campaign_id: int = 1) -> None:
        with db.open_db() as conn, db.transaction(conn):
            conn.execute(
                "INSERT INTO Measurements (trial, subject, protocol, env_image) "
                "VALUES ('trial_base', 1, 1, 'us-docker.pkg.dev/proj/vllm:latest')"
            )
            conn.execute(
                "UPDATE Campaigns SET state = 'Active', champion = 1, "
                "entry_baseline = 1, current_baseline = 1 WHERE id = ?",
                (campaign_id, ),
            )

    def _write_hyp(self, name: str = "hyp.json", **overrides: object) -> Path:
        path = self.base / name
        jsonio.write_json(
            path,
            {
                "lever_kind": "kernel",
                "target_scope": "step",
                "claim": "Double tile dimensions.",
                "short_claim": "Double tiles.",
                "evidence": [],
                "target_code_refs": ["model.py"],
                **overrides,
            },
        )
        return path
