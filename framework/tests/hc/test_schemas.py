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
"""Unit tests for `framework.hc.schemas` and `framework.hc.constants`."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from typing import Any

from framework.hc import constants, schemas

_BASE_PROTO_JSON = """{
  "workload_kind": "inference", "primary_metric": "tps", "compute_profile": "gke-v7x",
  "primary_metric_unit": "tok/s", "primary_metric_direction": "maximum",
  "primary_metric_statistic": "mean", "min_replicates": 3, "min_win_percentage": 2.0,
  "max_cell_regression_percentage": 3.0, "noise_floor_percentage": 1.0,
  "max_rounds": 10, "max_barren_rounds": 3, "target_improvement_percent": 20.0,
  "measurement_cells": [
    {"topology": "2x2x1", "weight": 0.6, "prefill": 1024, "decode": 1024, "concurrency": 64},
    {"topology": "2x2x1", "weight": 0.4, "prefill": 8192, "decode": 1024, "concurrency": 32}
  ],
  "accuracy_gate": {"metric": "gsm8k", "comparator": "gte", "threshold": -0.01,
                    "is_relative": true, "relative_to": null},
  "execution_spec": {"image": "img:1", "base_server_args": "--tp 4",
                     "base_client_args": "", "base_eval_args": "--num_fewshot 5"}
}"""


def _proto(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(_BASE_PROTO_JSON)
    data.update(copy.deepcopy(overrides))
    return data


def _hyp(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "lever_kind": "kernel",
        "target_scope": "step",
        "claim": "c",
        "short_claim": "s"
    }
    data.update(copy.deepcopy(overrides))
    return data


def _op(lk: str, op: str) -> dict[str, str]:
    return {"layer_kind": lk, "operator": op}


class TestSchemaValidation(unittest.TestCase):
    """Verify enums, error codes, and Campaign/Protocol/Hypothesis/flags validators."""

    def test_parse_enum_and_error_codes(self):
        v = constants.parse_enum(constants.DecisionVerdict, "Win", "verdict")
        self.assertEqual((v, str(v)), (constants.DecisionVerdict.WIN, "Win"))
        with self.assertRaises(constants.ValidationError) as ctx:
            constants.parse_enum(constants.DecisionVerdict, "win", "verdict")
        self.assertEqual(ctx.exception.error_code, "VALIDATION_ERROR")
        self.assertEqual(
            constants.InvalidStateError("x").error_code, "INVALID_STATE")
        p_props = schemas._load_hc_schema("hc-protocol.json")["properties"]
        h_props = schemas._load_hc_schema("hc-hypothesis.json")["properties"]
        self.assertEqual(p_props["workload_kind"]["enum"],
                         [m.value for m in constants.WorkloadKind])
        self.assertEqual(h_props["lever_kind"]["enum"],
                         [m.value for m in constants.LeverKind])

    def test_campaign_and_protocol_validation(self):
        cfg = schemas.validate_campaign_config(
            dict(name="m", target_model="g", engine="vllm", accelerator="v6e"))
        self.assertTrue(cfg["owner"])
        with self.assertRaises(constants.ValidationError):
            schemas.validate_campaign_config({**cfg, "extra_key": 1})

        padded = _proto(compute_profile="  gke-v7x\n")
        padded["execution_spec"]["base_eval_args"] = "  --num_fewshot 5  "
        valid = schemas.validate_protocol_config(padded)
        self.assertEqual(
            (valid["compute_profile"],
             valid["execution_spec"]["base_eval_args"]),
            ("gke-v7x", "--num_fewshot 5"),
        )
        micro_spec = dict(_proto()["execution_spec"],
                          microbenchmark_target="  plan.steps.json  ")
        micro_valid = schemas.validate_protocol_config(
            _proto(accuracy_gate=None, execution_spec=micro_spec),
            is_microbenchmark=True,
        )
        self.assertEqual(
            (micro_valid["accuracy_gate"],
             micro_valid["execution_spec"]["microbenchmark_target"]),
            (None, "plan.steps.json"),
        )

        tcell = dict(topology="2x2x2",
                     weight=1.0,
                     sequence_length=4096,
                     global_batch_size=128)
        self.assertEqual(
            schemas.validate_protocol_config(
                _proto(workload_kind="training",
                       measurement_cells=[tcell]))["workload_kind"],
            "training")

        icell, espec = _proto()["measurement_cells"][0], _proto(
        )["execution_spec"]
        bad_cases = [
            (_proto(measurement_cells=[dict(icell, weight=0.5)]),
             "sum to 1.0"),
            (_proto(noise_floor_percentage=2.5), "noise_floor_percentage"),
            (_proto(accuracy_gate=None), "accuracy_gate"),
            (_proto(measurement_cells=[dict(icell, sequence_length=4096)]),
             "forbids training"),
            (_proto(workload_kind="training"), "sequence_length"),
            (_proto(workload_kind="training",
                    measurement_cells=[dict(tcell,
                                            prefill=1)]), "forbids inference"),
            (_proto(
                execution_spec=dict(espec, microbenchmark_target="bench.py")),
             "only allowed on microbenchmark"),
            (_proto(execution_spec=dict(espec, base_eval_args="--limit 10")),
             "cannot specify '--limit'"),
            (_proto(execution_spec=dict(espec, base_eval_args="--tasks=mmlu")),
             "cannot specify '--tasks=mmlu'"),
            (_proto(execution_spec=dict(espec, base_eval_args="'unclosed")),
             "invalid shell quoting"),
        ]
        for bad_proto, err_re in bad_cases:
            with self.subTest(err_re=err_re), self.assertRaisesRegex(
                    constants.ValidationError, err_re):
                schemas.validate_protocol_config(bad_proto)

        for bad_micro_target in (None, "../evil.py", "bench.sh"):
            with self.assertRaises(constants.ValidationError):
                schemas.validate_protocol_config(
                    _proto(
                        accuracy_gate=None,
                        execution_spec=dict(
                            espec, microbenchmark_target=bad_micro_target),
                    ),
                    is_microbenchmark=True,
                )

        with tempfile.NamedTemporaryFile(suffix=".json") as bad_utf8:
            bad_utf8.write(b"\xff\xfe{}")
            bad_utf8.flush()
            with self.assertRaises(constants.ValidationError):
                schemas.load_json_object(bad_utf8.name, "Protocol Config")

    def test_hypothesis_and_flags_validation(self):
        self.assertEqual(
            schemas.validate_hypothesis_config(_hyp())["target_scope"], "step")
        ev = [dict(source_type="xprof", locator="t.pb", summary="42%")]
        hyp_ops = schemas.validate_hypothesis_config(
            _hyp(target_scope="canonical_ops",
                 target_canonical_ops=[_op("moe", "gmm")],
                 evidence=ev,
                 target_code_refs=["models/moe.py"]))
        self.assertEqual(len(hyp_ops["target_canonical_ops"]), 1)

        edges = [
            dict(producer=_op("dense", "qkv"), consumer=_op("dense", "attn")),
            dict(producer=_op("dense", "attn"), consumer=_op("dense", "mlp"))
        ]
        hyp_rel = schemas.validate_hypothesis_config(
            _hyp(lever_kind="sharding",
                 target_scope="redundant_relayout",
                 target_relayout_edges=edges))
        self.assertEqual(len(hyp_rel["target_relayout_edges"]), 2)

        bad_edges = [
            dict(producer=_op("dense", "a"), consumer=_op("dense", "b")),
            dict(producer=_op("moe", "c"), consumer=_op("moe", "d"))
        ]
        for bad_hyp, err_re in (
            (_hyp(target_scope="canonical_ops",
                  target_canonical_ops=[]), "non-empty"),
            (_hyp(lever_kind="sharding",
                  target_scope="redundant_relayout",
                  target_relayout_edges=bad_edges), "common operator"),
            (_hyp(lever_kind="sharding",
                  target_scope="redundant_relayout",
                  target_relayout_edges=[dict(producer=None,
                                              consumer=None)]), "non-null"),
            (_hyp(short_claim="x" * 501), "maxLength"),
        ):
            with self.subTest(err_re=err_re), self.assertRaisesRegex(
                    constants.ValidationError, err_re):
                schemas.validate_hypothesis_config(bad_hyp)

        self.assertEqual(
            schemas.format_canonical_flags(
                dict(b=1,
                     env=dict(XLA_FLAGS="--xla_a=1"),
                     server_args=["--tp", "4"])),
            '{\n  "b": 1,\n  "env": {\n    "XLA_FLAGS": "--xla_a=1"\n  },\n  "server_args": [\n    "--tp",\n    "4"\n  ]\n}\n',
        )
        for bf in (["bad"], {
                " ": 1
        }, {
                "env": ["b"]
        }, {
                "env": {
                    "1B": "x"
                }
        }, {
                "env": {
                    "OK": None
                }
        }, {
                "env": {
                    "OK": float("nan")
                }
        }, {
                "server_args": [1]
        }, {
                "a": float("inf")
        }, {
                "items": [float("-inf")]
        }, {
                "nested": {
                    "d": [1]
                }
        }, {
                "eval_limit": 1
        }, {
                "base_eval_args": "--num_fewshot 0"
        }, {
                "model": "tiny"
        }):
            with self.assertRaises(constants.ValidationError):
                schemas.validate_flags_dict(bf)


if __name__ == "__main__":
    unittest.main()
