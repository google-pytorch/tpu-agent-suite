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
"""Tests for the tpu-diagnostics request and result contract.

The contract types what the package owns and passes through what XProf owns. Most of
these tests pin the second half: XProf's tools, field names, status and reason
strings change between nightlies, and the pin is a floor, so a result the next
nightly produces must still validate.

A call reads any number of traces and may capture one more, so everything
measured lives on the trace it was measured on, and whether a trace was
captured is recorded on the trace rather than in the status.
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import unittest

_PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from framework import agent_contracts, jsonio, utils  # noqa: E402
from framework.builder import load_package  # noqa: E402
from framework.builder import resolve_manifest_composition  # noqa: E402

RUN_ID = "20260921T120000Z-abcdef012345"
REQUEST = "schemas/profiling-request.json"
RESULT = "schemas/profiling-result.json"

# Every instruction file that tells its agent how to call tpu-diagnostics.
CALLERS = [
    "skills/tpu-agent/SKILL.md",
    "agents/tpu-sharding.md",
    "agents/tpu-kernel-development.md",
    "agents/tpu-performance-optimization.md",
    "agents/tpu-performance-optimization-executor.md",
]

_JSON_BLOCK = re.compile(r"```json\n(.*?)\n\s*```", re.DOTALL)


def _trace(captured: bool = False, **metrics) -> dict:
    return {
        "path": "artifacts/3/tensorboard/plugins/profile/2026_09_21_12_30_00",
        "captured": captured,
        "metrics": metrics,
    }


def _kpi(value) -> dict:
    return {"value": value, "source_tool": "get_kpi_metrics"}


def _reported(*traces) -> dict:
    return {
        "status": "reported",
        "summary": "read it",
        "run_id": RUN_ID,
        "profiling_report_artifact": "artifacts/3/tpu_profiling_report.md",
        "xprof_version": "xprof 2.24.0",
        "traces": list(traces) or [_trace()],
    }


def _lever(**fields) -> dict:
    return {
        "kind": "kernel-block",
        "summary": "paged attention, decode",
        "share_pct": 21,
        "source_tool": "get_kernel_stats",
        **fields,
    }


def _ranked(*levers) -> dict:
    return _reported(dict(_trace(), levers=list(levers)))


def _flat(text: str) -> str:
    """Whitespace collapsed, so reflowing a paragraph doesn't break a check."""
    return " ".join(text.split())


def _routing_line(text: str, agent: str) -> str:
    """The coordinator's top-level bullet that delegates to `agent`."""
    for line in text.splitlines():
        if line.startswith("- ") and f"{{{{agent:{agent}}}}}" in line:
            return line
    raise AssertionError(f"no routing line delegates to {agent}")


class TestDiagnosticsSchemas(unittest.TestCase):
    """The profiling-request / profiling-result pair and who binds it."""

    @classmethod
    def setUpClass(cls):
        cls.request = jsonio.load_json(_PACKAGE_ROOT / REQUEST)
        cls.result = jsonio.load_json(_PACKAGE_ROOT / RESULT)

    def _result_errors(self, payload: dict) -> list[str]:
        """Both halves the SubagentStop hook applies: shape, then status."""
        return (utils.validate_instance(payload, self.result) +
                agent_contracts.check_status_payload(self.result, payload))

    # -- registration and binding --------------------------------------------

    def test_schemas_are_registered(self):
        """Every schemas/*.json is registered; package.release.json lists none."""
        registered = set(load_package(_PACKAGE_ROOT)["schemas"])
        for rel in (REQUEST, RESULT):
            with self.subTest(schema=rel):
                self.assertTrue((_PACKAGE_ROOT / rel).is_file())
                self.assertIn(rel, registered)

    def test_schema_definitions_are_structurally_valid(self):
        for rel, schema in ((REQUEST, self.request), (RESULT, self.result)):
            with self.subTest(schema=rel):
                self.assertEqual(utils.validate_schema_definition(schema), [])

    def test_examples_validate(self):
        for rel, schema in ((REQUEST, self.request), (RESULT, self.result)):
            examples = schema.get("examples", [])
            self.assertTrue(examples, f"{rel} has no examples")
            for i, example in enumerate(examples):
                with self.subTest(schema=rel, example=i):
                    self.assertEqual(utils.validate_instance(example, schema),
                                     [])
                    self.assertEqual(
                        agent_contracts.check_status_payload(schema, example),
                        [])

    def test_examples_cover_a_capture_and_a_comparison(self):
        """The two shapes a single `trace_path` could not carry."""
        traces = [
            e.get("traces", []) for e in self.result["examples"]
            if e["status"] == "reported"
        ]
        self.assertTrue(any(t and t[0]["captured"] for t in traces))
        self.assertTrue(any(len(t) >= 2 for t in traces))

    def test_tpu_diagnostics_binds_the_pair(self):
        manifest = resolve_manifest_composition(
            _PACKAGE_ROOT,
            jsonio.load_json(_PACKAGE_ROOT / "package.release.json"))
        agent = {a["name"]: a for a in manifest["agents"]}["tpu-diagnostics"]
        self.assertEqual(agent.get("accepts"), REQUEST)
        self.assertEqual(agent.get("produces"), RESULT)

    def test_instructions_show_the_json_they_promise(self):
        """A produces-bound agent told to answer in prose is denied every time."""
        text = (_PACKAGE_ROOT /
                "agents/tpu-diagnostics.md").read_text(encoding="utf-8")
        _, _, body = text.partition("\n---\n")
        self.assertIn("```json", body)
        block = json.loads(_JSON_BLOCK.search(body).group(1).strip())
        self.assertEqual(block["status"], "reported")
        block["run_id"] = RUN_ID  # the placeholder the agent fills
        self.assertEqual(self._result_errors(block), [])
        self.assertIn("levers", block["traces"][0])

    def test_every_failure_reason_is_explained_to_the_agent(self):
        """The enum is package-owned, so the agent must be told when each applies."""
        text = (_PACKAGE_ROOT /
                "agents/tpu-diagnostics.md").read_text(encoding="utf-8")
        for reason in self.result["properties"]["failure_reason"]["enum"]:
            with self.subTest(reason=reason):
                self.assertIn(f"`{reason}`", text)

    def test_every_lever_kind_is_explained_to_the_agent(self):
        """The agent picks the kind, so it must be told what each one means."""
        text = (_PACKAGE_ROOT /
                "agents/tpu-diagnostics.md").read_text(encoding="utf-8")
        trace = self.result["properties"]["traces"]["items"]
        lever = trace["properties"]["levers"]["items"]
        for kind in lever["properties"]["kind"]["enum"]:
            with self.subTest(kind=kind):
                self.assertIn(f"`{kind}`", text)

    # -- the request: prose and programmatic callers alike -------------------

    def test_request_accepts_a_prompt_and_run_id_alone(self):
        """tpu-agent delegates conversationally; that must not be denied."""
        self.assertEqual(
            utils.validate_instance(
                {
                    "prompt": "Where does the time go?",
                    "run_id": RUN_ID
                }, self.request), [])

    def test_request_accepts_any_callers_field_names(self):
        """Sharding and kernel ask for different quantities; neither is special."""
        for fields in (["duty_cycle_percent", "tc_idle_ms_average"], [
                "alu_utilization_percent", "dma_sync_wait_to_compute_ratio"
        ], ["a_metric_no_xprof_version_has_yet"]):
            with self.subTest(fields=fields):
                self.assertEqual(
                    utils.validate_instance(
                        {
                            "prompt": "x",
                            "run_id": RUN_ID,
                            "traces": [{
                                "path": "gs://bucket/run/tensorboard"
                            }],
                            "fields": fields,
                        }, self.request), [])

    def test_request_carries_a_comparison_and_a_before_after(self):
        for label, payload in [
            ("two labelled traces", {
                "traces": [{
                    "path": "/p/a",
                    "label": "A"
                }, {
                    "path": "/p/b",
                    "label": "B",
                    "session_id": "2026_08_18_09_12_03"
                }]
            }),
            ("a trace and a workload", {
                "traces": [{
                    "path": "/p/baseline",
                    "label": "baseline"
                }],
                "workload": "/w/variant.py"
            }),
            ("a workload alone", {
                "workload": "/w/variant.py"
            }),
        ]:
            with self.subTest(case=label):
                self.assertEqual(
                    utils.validate_instance(
                        dict(payload, prompt="x", run_id=RUN_ID),
                        self.request), [])

    def _server_request(self, **server) -> dict:
        return {
            "prompt": "Rank the decode bottlenecks.",
            "run_id": RUN_ID,
            "server": {
                "job_id": "gpt-oss-serve-1a2b",
                "port": 8000,
                **server
            },
        }

    def test_request_carries_the_servers_configuration(self):
        """The launch files reach the profiler as paths, not a prose summary."""
        for label, serving_config in [
            ("both files", {
                "deployments_json": "/r/inference/deployments.json",
                "launch_recipe": "/r/inference/launch_vllm.sh",
            }),
            ("manifest only", {
                "deployments_json": "/r/inference/deployments.json"
            }),
        ]:
            with self.subTest(case=label):
                self.assertEqual(
                    utils.validate_instance(
                        self._server_request(serving_config=serving_config),
                        self.request), [])
        for label, serving_config in [
            ("no manifest", {
                "launch_recipe": "/r/inference/launch_vllm.sh"
            }),
            ("empty manifest", {
                "deployments_json": ""
            }),
            ("unknown key", {
                "deployments_json": "/r/d.json",
                "max_model_len": 8192
            }),
        ]:
            with self.subTest(case=label):
                self.assertTrue(
                    utils.validate_instance(
                        self._server_request(serving_config=serving_config),
                        self.request))

    def test_profiler_and_kernel_take_the_same_serving_config(self):
        """tpu-agent copies one onboarding object into both requests."""
        kernel = jsonio.load_json(
            _PACKAGE_ROOT /
            "schemas/kernel-request.json")["properties"]["serving_config"]
        profiler = self.request["properties"]["server"]["properties"][
            "serving_config"]
        for key in ("type", "additionalProperties", "required"):
            with self.subTest(key=key):
                self.assertEqual(profiler[key], kernel[key])
        self.assertEqual(
            {
                k: {
                    f: v
                    for f, v in p.items() if f != "description"
                }
                for k, p in profiler["properties"].items()
            }, {
                k: {
                    f: v
                    for f, v in p.items() if f != "description"
                }
                for k, p in kernel["properties"].items()
            })

    def test_tpu_agent_suite_and_profiler_use_the_launch_files(self):
        tpu_agent_suite = _flat(
            (_PACKAGE_ROOT /
             "skills/tpu-agent/SKILL.md").read_text(encoding="utf-8"))
        profiler = _flat(
            (_PACKAGE_ROOT /
             "agents/tpu-diagnostics.md").read_text(encoding="utf-8"))
        for text, phrase in [
            (tpu_agent_suite, "and `serving_config` with the "
             "`output_files.deployments_json` and "
             "`output_files.launch_recipe` paths"),
            (tpu_agent_suite,
             "Only for a server tpu-model-onboarding did not launch"),
            (profiler, "When the request carries `server.serving_config`, "
             "read both files"),
            (profiler, "never restart or relaunch the server from them"),
            (profiler, "read from the launch recipe in "
             "`server.serving_config`"),
        ]:
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_request_rejects_malformed_payloads(self):
        for label, payload in [
            ("no run_id", {
                "prompt": "x"
            }),
            ("malformed run_id", {
                "prompt": "x",
                "run_id": "not-a-run-id"
            }),
            ("unknown field", {
                "prompt": "x",
                "run_id": RUN_ID,
                "capture_needed": True
            }),
            ("the retired single trace_path", {
                "prompt": "x",
                "run_id": RUN_ID,
                "trace_path": "/p/a"
            }),
            ("a trace without a path", {
                "prompt": "x",
                "run_id": RUN_ID,
                "traces": [{
                    "label": "A"
                }]
            }),
            ("an empty trace path", {
                "prompt": "x",
                "run_id": RUN_ID,
                "traces": [{
                    "path": ""
                }]
            }),
            ("an unknown key on a trace", {
                "prompt": "x",
                "run_id": RUN_ID,
                "traces": [{
                    "path": "/p/a",
                    "captured": True
                }]
            }),
            ("negative top_ops", {
                "prompt": "x",
                "run_id": RUN_ID,
                "top_ops": -1
            }),
        ]:
            with self.subTest(case=label):
                self.assertTrue(utils.validate_instance(payload, self.request))

    def test_every_documented_caller_request_validates(self):
        """A caller's example is what its agent copies; it must pass the hook."""
        checked = 0
        for rel in CALLERS + ["README.md"]:
            text = (_PACKAGE_ROOT / rel).read_text(encoding="utf-8")
            for block in _JSON_BLOCK.findall(text):
                if '"traces"' not in block and '"workload"' not in block:
                    continue  # some other agent's request
                payload = json.loads(block.strip())
                payload["run_id"] = RUN_ID  # the placeholder the caller fills
                checked += 1
                with self.subTest(caller=rel):
                    self.assertEqual(
                        utils.validate_instance(payload, self.request), [])
        self.assertGreaterEqual(checked, 3)

    def test_every_caller_names_the_request_contract(self):
        for rel in CALLERS:
            with self.subTest(caller=rel):
                text = (_PACKAGE_ROOT / rel).read_text(encoding="utf-8")
                self.assertIn("profiling-request", text)
                self.assertNotIn('"trace_path"', text)

    # -- the result: pass XProf through, type what the package owns ----------

    def test_result_passes_xprof_values_through_untouched(self):
        """Every shape a current nightly returns for a KPI must validate."""
        self.assertEqual(
            self._result_errors(
                _reported(
                    _trace(
                        duty_cycle_percent={
                            "value": "99.4%",
                            "source_tool": "get_kpi_metrics",
                            "unit": "%"
                        },
                        # The tool's own string when there is no memory data.
                        peak_hbm_gib={
                            "value": "N/A",
                            "source_tool": "get_kpi_metrics"
                        },
                        # An object, not a number.
                        accelerator_info={
                            "value": {
                                "device_type": "TPU v6 Lite",
                                "device_core_count": "4"
                            },
                            "source_tool": "get_kpi_metrics"
                        },
                        llo_present={
                            "value": False,
                            "source_tool": "get_llo_analysis"
                        },
                        roofline_utilization={
                            "value": "49.33%",
                            "source_tool": "get_kpi_metrics",
                            "provenance": "opaque_custom_call"
                        },
                        # What the live agent did unprompted: explain the value.
                        mxu_utilization_percent={
                            "value":
                            "26.7%",
                            "source_tool":
                            "get_kpi_metrics",
                            "note":
                            "Lower bound: 20 custom calls carry no"
                            " cost model.",
                        },
                        sc_idle_ms_average={
                            "value": None,
                            "source_tool": "none"
                        },
                    ))),
            [])

    def test_result_accepts_status_and_reason_strings_xprof_has_not_shipped(
            self):
        """Upstream adds statuses and reasons; an enum here would deny them."""
        self.assertEqual(
            self._result_errors(
                _reported(
                    _trace(
                        alu_utilization_percent={
                            "value": None,
                            "source_tool": "get_llo_analysis",
                            "tool_status": "UNAVAILABLE",
                            "tool_reason": "SOME_REASON_FROM_A_FUTURE_NIGHTLY",
                        }))), [])

    def test_unavailable_llo_data_is_reported_not_failed(self):
        """The instructions say so; the schema must allow it without a reason."""
        payload = _reported(
            _trace(
                alu_utilization_percent={
                    "value": None,
                    "source_tool": "get_llo_analysis",
                    "tool_status": "UNAVAILABLE",
                    "tool_reason": "LLO_DATA_ABSENT",
                }))
        self.assertNotIn("failure_reason", payload)
        self.assertEqual(self._result_errors(payload), [])

    def test_result_carries_one_set_of_metrics_per_trace(self):
        """A comparison: the same field, two values, neither overwriting."""
        payload = _reported(
            dict(_trace(step_time_ms=_kpi("233.82")), label="A"),
            dict(_trace(step_time_ms=_kpi("312.40")), label="B"))
        self.assertEqual(self._result_errors(payload), [])

    def test_result_mixes_a_supplied_and_a_captured_trace(self):
        """A before/after: the baseline read as given, the variant captured."""
        payload = _reported(dict(_trace(captured=False), label="baseline"),
                            dict(_trace(captured=True), label="variant"))
        self.assertEqual(self._result_errors(payload), [])

    def test_a_hot_op_row_copies_into_a_valid_fusion_context(self):
        """tpu-agent hands a row to tpu-kernel-development as `fusion_context`.

        kernel-request requires `fusion` and `fused_with`, so a row allowed
        to omit them could be copied into a request the boundary refuses.
        """
        fusion_context = jsonio.load_json(
            _PACKAGE_ROOT /
            "schemas/kernel-request.json")["properties"]["fusion_context"]
        rows = [
            row for example in self.result["examples"]
            for trace in example.get("traces", [])
            for row in trace.get("hot_ops", [])
        ]
        self.assertTrue(rows, "no example carries a hot_ops row")
        for row in rows:
            copied = {
                key: row[key]
                for key in ("fusion", "fused_with", "source", "per_call_us")
                if key in row
            }
            copied["profiling_report"] = "/runs/r/artifacts/report.md"
            with self.subTest(op=row["trace_name"]):
                self.assertEqual(
                    utils.validate_instance(copied, fusion_context), [])

    def test_a_hot_op_rows_call_shapes_copy_into_a_served_call(self):
        """tpu-agent hands `call_shapes` to tpu-kernel-development as `served_calls`."""
        served_calls = jsonio.load_json(
            _PACKAGE_ROOT /
            "schemas/kernel-request.json")["properties"]["served_calls"]
        rows = [
            row for example in self.result["examples"]
            for trace in example.get("traces", [])
            for row in trace.get("hot_ops", []) if "call_shapes" in row
        ]
        self.assertTrue(rows, "no example carries a row with call_shapes")
        for row in rows:
            case = {
                "label": "decode",
                "call_shapes": row["call_shapes"],
                "per_call_us": row["per_call_us"],
            }
            with self.subTest(op=row["trace_name"]):
                self.assertEqual(
                    utils.validate_instance({"cases": [case]}, served_calls),
                    [])

    def test_a_server_captures_load_copies_into_a_served_call(self):
        """What the server did under the capture reaches the harness as data.

        A trace shows each call's shapes, not how many sequences were live;
        the profiler records that from the server's metrics, and tpu-agent
        copies it into the served-call case unchanged.
        """
        served_calls = jsonio.load_json(
            _PACKAGE_ROOT /
            "schemas/kernel-request.json")["properties"]["served_calls"]
        row = {
            "trace_name": "ragged_paged_attention",
            "source_tool": "get_top_hlo_ops",
            "per_call_us": 45.26,
            "fusion": None,
            "fused_with": [],
            "call_shapes": {
                "operands": ["s32[256]{0}", "bf16[8,16,4,2,128]{4,3,2,1,0}"],
                "outputs": ["bf16[8,16,4,2,128]{4,3,2,1,0}"],
            },
        }
        trace_load = {
            "sent": "512 requests, in/out 2048/256, concurrency 32",
            "runtime_values": {
                "requests_running": {
                    "min": 4,
                    "median": 28,
                    "max": 32
                },
            },
        }
        trace = dict(_trace(captured=True), hot_ops=[row], load=trace_load)
        self.assertEqual(self._result_errors(_reported(trace)), [])
        case = {
            "label": "decode",
            "call_shapes": row["call_shapes"],
            "per_call_us": row["per_call_us"],
            "load": trace_load["sent"],
            "runtime_values": trace_load["runtime_values"],
        }
        self.assertEqual(
            utils.validate_instance({"cases": [case]}, served_calls), [])

    def test_every_call_to_the_levers_kernel_copies_into_a_served_call(self):
        """One case per call to the kernel, not one per trace.

        In the gpt-oss run one capture held six calls to gmm_v2 under six
        trace names; picked by the lever's `trace_name`, only the decode
        gate/up call reached tpu-kernel-development, and the harness took the other five
        from the report's prose. `source_file` ties them together.
        """
        served_calls = jsonio.load_json(
            _PACKAGE_ROOT /
            "schemas/kernel-request.json")["properties"]["served_calls"]
        gmm = "vllm_torchtpu/kernels/megablox/gmm_v2.py"

        def row(name, per_call_us, operands, outputs, **fields):
            return {
                "trace_name": name,
                "source_tool": "get_kernel_stats",
                "per_call_us": per_call_us,
                "fusion": None,
                "fused_with": [],
                "call_shapes": {
                    "operands": operands,
                    "outputs": outputs
                },
                **fields,
            }

        rows = [
            row("gmm_v2-g_32-m_64-k_2944-act_swigluoai-n_768",
                56.17, ["bf16[64,2944]", "bf16[32,2944,768]"],
                ["bf16[64,384]"],
                source_file=gmm),
            row("gmm_v2-g_32-m_64-k_384-act_None-n_2944",
                32.06, ["bf16[64,384]", "bf16[32,384,2944]"],
                ["bf16[64,2944]"],
                source_file=gmm),
            row("gmm_v2-g_32-m_8192-k_2944-act_swigluoai-n_768",
                152.69, ["bf16[8192,2944]", "bf16[32,2944,768]"],
                ["bf16[8192,384]"],
                source_file=gmm),
            row("RPA-HD_64-bq_16-bkvp_16-p_256",
                22.87, ["bf16[1,16,4,2,128]"], ["bf16[1,16,4,2,128]"],
                source_file="vllm_torchtpu/kernels/ragged_paged_attention/"
                "v3/kernel_hd64.py"),
        ]
        levers = [
            _lever(trace_name=rows[0]["trace_name"],
                   source_file=gmm,
                   phase="decode",
                   share_pct=34.11),
            _lever(trace_name=rows[1]["trace_name"],
                   source_file=gmm,
                   phase="decode",
                   share_pct=19.47),
        ]
        trace = dict(_trace(captured=True),
                     label="c16-1024in-128out",
                     hot_ops=rows,
                     levers=levers)
        self.assertEqual(self._result_errors(_reported(trace)), [])

        # tpu-agent's rule: every row whose source_file is the lever's.
        lever = levers[0]
        by_name = {lv["trace_name"]: lv for lv in trace["levers"]}
        cases = []
        for r in trace["hot_ops"]:
            if r.get("source_file") != lever["source_file"]:
                continue
            case = {
                "label": f"{trace['label']} / {r['trace_name']}",
                "call_shapes": r["call_shapes"],
                "per_call_us": r["per_call_us"],
            }
            if r["trace_name"] in by_name:
                case["phase"] = by_name[r["trace_name"]]["phase"]
                case["share_pct"] = by_name[r["trace_name"]]["share_pct"]
            cases.append(case)
        self.assertEqual([c["per_call_us"] for c in cases],
                         [56.17, 32.06, 152.69])
        self.assertNotIn("share_pct", cases[2])
        self.assertEqual(
            utils.validate_instance({"cases": cases}, served_calls), [])

    def test_a_traces_load_owes_what_was_sent(self):
        for label, trace_load in [
            ("no sent", {
                "runtime_values": {}
            }),
            ("empty sent", {
                "sent": ""
            }),
            ("unknown key", {
                "sent": "8 requests",
                "live_seqs": 4
            }),
        ]:
            with self.subTest(case=label):
                trace = dict(_trace(captured=True), load=trace_load)
                self.assertTrue(self._result_errors(_reported(trace)))

    def test_profiler_keeps_the_layout_and_reads_server_metrics(self):
        """The two values a harness cannot get back once the profiler drops them."""
        text = _flat((_PACKAGE_ROOT /
                      "agents/tpu-diagnostics.md").read_text(encoding="utf-8"))
        for phrase in ("with the layout suffix in braces whenever the text "
                       "prints one", "`GET /metrics` on the server's port",
                       "do not fill it from the load you sent",
                       "Give every custom call its kernel file"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_tpu_agent_suite_passes_every_artifact_as_typed_data(self):
        """Profiler and onboarding artifacts reach tpu-kernel-development unrewritten."""
        text = (_PACKAGE_ROOT /
                "skills/tpu-agent/SKILL.md").read_text(encoding="utf-8")
        after = text.partition("\n## After a profile")[2]
        section = _flat(after.partition("\n## ")[0])
        for phrase in ("copied verbatim from the result that produced it",
                       "`fusion_context`, from the `hot_ops` row",
                       "`served_calls`: one case per `hot_ops` row",
                       "each row whose `source_file` is the lever's",
                       "`call_shapes` and `per_call_us`",
                       "`load.runtime_values` as `runtime_values`",
                       "never restate them in the `prompt`",
                       "`serving_config`: the `output_files.deployments_json`",
                       "including option 2 below"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, section)

    def test_a_hot_op_row_owes_its_fusion(self):
        """Unfused is `null` and `[]`, never a missing key."""
        row = {
            "trace_name": "fusion.668",
            "source_tool": "get_top_hlo_ops",
            "fusion": None,
            "fused_with": [],
        }
        self.assertEqual(
            self._result_errors(_reported(dict(_trace(), hot_ops=[row]))), [])
        for key in ("fusion", "fused_with"):
            with self.subTest(missing=key):
                partial = {k: v for k, v in row.items() if k != key}
                self.assertTrue(
                    self._result_errors(
                        _reported(dict(_trace(), hot_ops=[partial]))))

    def test_result_rejects_what_package_owns_when_malformed(self):
        base = _reported()
        for label, payload in [
            ("metric without source_tool", _reported(_trace(x={"value": 1}))),
            ("metric without value",
             _reported(_trace(x={"source_tool": "t"}))),
            ("unknown key on a metric",
             _reported(
                 _trace(x={
                     "value": 1,
                     "source_tool": "t",
                     "estimate": 2
                 }))),
            ("trace without captured", dict(base, traces=[{
                "path": "/p/a"
            }])),
            ("trace without path", dict(base, traces=[{
                "captured": False
            }])),
            ("unknown key on a trace",
             dict(base, traces=[dict(_trace(), top_ops=[])])),
            ("tool output without artifact",
             dict(base,
                  traces=[
                      dict(_trace(), tool_outputs=[{
                          "tool": "get_overview"
                      }])
                  ])),
            ("lever without kind",
             _ranked({
                 "summary": "x",
                 "share_pct": 5,
                 "source_tool": "t"
             })),
            ("lever without share_pct",
             _ranked({
                 "kind": "other",
                 "summary": "x",
                 "source_tool": "t"
             })),
            ("lever kind outside the enum",
             _ranked(_lever(kind="slow-matmul"))),
            ("share_pct above 100", _ranked(_lever(share_pct=126))),
            ("phase outside the enum", _ranked(_lever(phase="steady"))),
            ("unknown key on a lever", _ranked(_lever(speedup=1.18))),
            ("the retired top-level metrics", dict(base, metrics={})),
            ("the retired top-level trace_path", dict(base,
                                                      trace_path="/p/a")),
            ("the retired status", dict(base, status="profiled")),
            ("failure_reason outside the enum",
             dict(base, status="failed", failure_reason="it_broke")),
            ("unknown status", dict(base, status="partial")),
            ("malformed run_id", dict(base, run_id="x")),
        ]:
            with self.subTest(case=label):
                self.assertTrue(self._result_errors(payload))

    def test_xprofs_envelope_is_never_a_metric_of_its_own(self):
        """What the live v6e agent did: status and provenance as metric names.

        A caller reading `traces[0].metrics` would find a metric called
        `tool_status`. Which fields were requested is beyond this schema, but
        the metric wrapper's own key names can be refused outright.
        """
        for name in ("tool_status", "tool_reason", "provenance", "note",
                     "unit", "value", "source_tool"):
            with self.subTest(name=name):
                self.assertTrue(
                    self._result_errors(
                        _reported(
                            _trace(
                                **{
                                    name: {
                                        "value": "NO_DATA",
                                        "source_tool": "get_step_trace"
                                    }
                                }))))

    def test_reported_owes_the_report_the_version_and_the_traces(self):
        for missing in ("profiling_report_artifact", "xprof_version",
                        "traces"):
            payload = _reported()
            payload.pop(missing)
            with self.subTest(missing=missing):
                self.assertTrue(self._result_errors(payload))

    def test_reported_with_no_traces_is_denied(self):
        """No `minItems` in this validator; `status_requires` treats [] as absent."""
        self.assertTrue(self._result_errors(dict(_reported(), traces=[])))

    def test_failed_owes_a_reason(self):
        self.assertTrue(
            self._result_errors({
                "status": "failed",
                "summary": "x",
                "run_id": RUN_ID
            }))

    def test_unhappy_paths_need_no_deliverables(self):
        """A run that read nothing has no report, version or trace to give."""
        for payload in (
            {
                "status": "invalid_input",
                "summary": "no traces and no workload",
                "run_id": RUN_ID
            },
            {
                "status": "invalid_input",
                "summary": "nothing at that path",
                "run_id": RUN_ID,
                "failure_reason": "trace_not_found",
                "error_message": "no .xplane.pb under /w/3/tensorboard",
            },
            {
                "status": "needs_input",
                "summary": "the workload does not construct its inputs",
                "run_id": RUN_ID
            },
                *({
                    "status": "failed",
                    "summary": "x",
                    "run_id": RUN_ID,
                    "failure_reason": reason,
                    "error_message": "tool output",
                } for reason in ("trace_empty_or_partial", "capture_failed",
                                 "xprof_unavailable", "read_timeout")),
        ):
            with self.subTest(status=payload["status"],
                              reason=payload.get("failure_reason")):
                self.assertEqual(self._result_errors(payload), [])

    def test_failed_may_keep_the_traces_it_did_read(self):
        """One of two traces unreadable: fail, but do not throw away the other."""
        self.assertEqual(
            self._result_errors({
                "status": "failed",
                "summary": "B is empty; A was read",
                "run_id": RUN_ID,
                "failure_reason": "trace_empty_or_partial",
                "traces": [dict(_trace(), label="A")],
            }), [])

    # -- the coordinator: route on the first lever ---------------------------

    def test_routing_lines_defer_to_after_a_profile(self):
        """The ranking picks the specialist, not the wording of the ask."""
        text = (_PACKAGE_ROOT /
                "skills/tpu-agent/SKILL.md").read_text(encoding="utf-8")
        for agent in ("tpu-diagnostics", "tpu-kernel-development"):
            with self.subTest(agent=agent):
                self.assertIn("After a profile", _routing_line(text, agent))

    def test_tpu_agent_suite_routes_on_the_first_lever(self):
        """tpu-kernel-development goes first only if the top lever is a `kernel-block`."""
        text = (_PACKAGE_ROOT /
                "skills/tpu-agent/SKILL.md").read_text(encoding="utf-8")
        after = text.partition("\n## After a profile")[2]
        section = _flat(after.partition("\n## ")[0])
        for phrase in ("Route on the first lever",
                       "`source_file` as `base_kernel_path`",
                       "don't start with tpu-kernel-development",
                       "{{built-in:ask_user_question}}",
                       "{{agent:tpu-model-onboarding}}",
                       "If nobody can answer", "wasn't worked on"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, section)

    def test_profiler_instructions_implement_prescribed_resolution(self):
        """Profiler locates and copies the served kernel through compute."""
        text = _flat((_PACKAGE_ROOT /
                      "agents/tpu-diagnostics.md").read_text(encoding="utf-8"))
        for phrase in (
                "{{script:tpu-diagnostics/introspect_kernel.py}}",
                "never with raw cluster commands",
                "--colocate-with <server.job_id>",
                "--output served_kernel.json --output served_kernel.tar.gz",
                "-- --dispatcher <file> --image <image>",
                "job wait --job-id <id> --until done",
                "installs or mounts no code at startup",
                "served_kernel",
                "`STAGING_NOT_CONFIGURED` or `missing_outputs`",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)
        self.assertNotIn("guaranteed to be the code the server runs", text)
        self.assertIn("scripts/tpu-diagnostics/introspect_kernel.py", text)

    def test_profiler_matches_trace_name_for_every_workload(self):
        """Serving must not bypass the exact trace_name match."""
        text = _flat((_PACKAGE_ROOT /
                      "agents/tpu-diagnostics.md").read_text(encoding="utf-8"))
        start = text.index("- `trace_name` and `source_file`, for a "
                           "`kernel-block`:")
        bullet = text[start:text.index("- `next_step`:", start)]
        for phrase in (
                "decided the same way for every workload, local or served",
                "whose name format produces `trace_name` exactly",
                "`RPAd-p256-…` and `RPAd-p_256-…` come from different kernels",
                "Launch flags only break ties",
                "--dispatcher <file>",
                "choose the kernel from the copies as above",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, bullet)
        self.assertNotIn("For local runs without containers", bullet)
        # A concrete package path in the instructions gets copied verbatim.
        self.assertNotIn("vllm_torchtpu.kernels", bullet)
        self.assertNotIn("batched_rpa", bullet)

    def test_tpu_agent_asks_diagnostics_for_source_file_once(self):
        text = _flat((_PACKAGE_ROOT /
                      "skills/tpu-agent/SKILL.md").read_text(encoding="utf-8"))
        for phrase in (
                "Ask `tpu-diagnostics` once to introspect",
                "stop and report its `summary`",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_served_kernel_copy_is_never_patched_to_import(self):
        """The copied kernel is read-only; a missing import is reported."""

        def text_of(relative):
            return _flat(
                (_PACKAGE_ROOT / relative).read_text(encoding="utf-8"))

        expected = {
            "agents/tpu-diagnostics.md": (
                "every file unchanged, nothing added",
                "never add `__init__.py` files or symlinks to make it import",
                "each marked present or absent in the copy",
                "never supply it",
            ),
            "agents/tpu-model-onboarding.md": (
                "Never write a file under a third-party package's name",
                "never symlink one package name to another to make an import "
                "resolve",
                "report it as missing",
            ),
            "skills/tpu-agent/SKILL.md": (
                "Pass `source_file` exactly as given",
                "Never ask an agent for a kernel's files by name from "
                "documentation or a public-repo listing",
            ),
            "agents/kernel-worker.md": (
                "If mock execution failed because an import of `base.py`'s own "
                "package did not resolve",
                "Stop before Phase 1 and return `invalid_input`",
                "only `base.py`'s own edits take effect",
            ),
            "agents/tpu-kernel-development.md": (
                "The base kernel's imports don't resolve on the target",
                "**Do not re-dispatch**, and never patch around it",
            ),
        }
        for relative, phrases in expected.items():
            text = text_of(relative)
            for phrase in phrases:
                with self.subTest(file=relative, phrase=phrase):
                    self.assertIn(phrase, text)
        schema = (_PACKAGE_ROOT /
                  "schemas/profiling-result.json").read_text(encoding="utf-8")
        self.assertIn("so only this file's edits take effect", schema)

    def test_served_kernel_lands_directly_under_work(self):
        """`<work>` is already `<artifacts>/tpu-diagnostics`: no doubled path."""
        for relative in ("agents/tpu-diagnostics.md",
                         "schemas/profiling-result.json"):
            text = (_PACKAGE_ROOT / relative).read_text(encoding="utf-8")
            with self.subTest(file=relative):
                self.assertIn("<work>/served_kernel/", text)
                self.assertNotIn("<work>/artifacts/", text)

    def test_introspection_image_comes_from_the_request_or_profile(self):
        schema = json.loads(
            (_PACKAGE_ROOT /
             "schemas/profiling-request.json").read_text(encoding="utf-8"))
        server = schema["properties"]["server"]
        self.assertIn("image", server["properties"])
        self.assertNotIn("image", server["required"])
        text = _flat((_PACKAGE_ROOT /
                      "agents/tpu-diagnostics.md").read_text(encoding="utf-8"))
        for phrase in (
                "the request's `server.image` when it carries one, otherwise "
                "the compute profile's `image`",
                "When the request carries `server.image`, also pass "
                "`--image <server.image>` to `compute submit`",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)
        skill = _flat(
            (_PACKAGE_ROOT /
             "skills/tpu-agent/SKILL.md").read_text(encoding="utf-8"))
        self.assertIn(
            "`image` only when the server runs an image other than the "
            "compute profile's", skill)

    def test_tpu_agent_kernel_block_cases_do_not_overlap(self):
        """An in-image kernel goes to diagnostics, never to a reference."""
        text = _flat((_PACKAGE_ROOT /
                      "skills/tpu-agent/SKILL.md").read_text(encoding="utf-8"))
        start = text.index("**Compiler-generated op, no kernel file**")
        compiler_case = text[start:text.index("**Custom call whose file", start
                                              )]
        custom_case = text[text.index("**Custom call whose file"):]
        custom_case = custom_case[:custom_case.index("don't ask a third time")]
        self.assertIn("is not a custom call", compiler_case)
        self.assertIn("extract_kernel_reference.py", compiler_case)
        self.assertIn("Never use this for a custom call", compiler_case)
        self.assertNotIn("site-packages", compiler_case)
        self.assertNotIn("/workspace", compiler_case)
        self.assertIn("site-packages", custom_case)
        self.assertIn("tpu-diagnostics", custom_case)
        self.assertNotIn("extract_kernel_reference.py", custom_case)


if __name__ == "__main__":
    unittest.main()
