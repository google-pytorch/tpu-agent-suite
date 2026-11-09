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
"""Validation unit tests for the tpu-kernel-development request: `input_language` and `served_calls`."""

import pathlib
import sys
import unittest

# Ensure repository root directory is in sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from framework import agent_contracts, jsonio, utils  # noqa: E402

PACKAGE_DIR = REPO_ROOT / "package"
RUN_ID = "20260914T120000Z-000000000000"


def _request(**fields):
    return {
        "prompt": "Make vLLM's ragged paged attention kernel faster.",
        "run_id": RUN_ID,
        "base_kernel_path": "work/ragged_paged_attention.py",
        **fields,
    }


class TestKernelRequestInputLanguage(unittest.TestCase):
    """`input_language` accepts the label callers reach for first."""

    @classmethod
    def setUpClass(cls):
        cls.request_schema = jsonio.load_json(PACKAGE_DIR / "schemas" /
                                              "kernel-request.json")
        cls.result_schema = jsonio.load_json(PACKAGE_DIR / "schemas" /
                                             "kernel-result.json")

    def test_accepts_pallas(self):
        """An existing Pallas kernel can be named as `pallas`."""
        self.assertEqual(
            utils.validate_instance(_request(input_language="pallas"),
                                    self.request_schema), [])

    def test_still_accepts_existing_languages(self):
        for language in ("jax", "pytorch", "specification"):
            with self.subTest(language=language):
                self.assertEqual(
                    utils.validate_instance(_request(input_language=language),
                                            self.request_schema), [])

    def test_rejects_cuda_and_unknown_languages(self):
        """CUDA stays reference-only; other labels are still refused."""
        for language in ("cuda", "triton", "Pallas"):
            with self.subTest(language=language):
                self.assertTrue(
                    utils.validate_instance(_request(input_language=language),
                                            self.request_schema))

    def test_examples_validate(self):
        for example in self.request_schema.get("examples", []):
            self.assertEqual(
                utils.validate_instance(example, self.request_schema), [])

    def test_result_never_reports_pallas(self):
        """`pallas` is recorded as `jax`, so the result has no such value."""
        result_language = self.result_schema["properties"]["input_language"]
        self.assertEqual(utils.validate_instance("jax", result_language), [])
        self.assertTrue(utils.validate_instance("pallas", result_language))

    def test_orchestrator_records_pallas_as_jax(self):
        """tpu-kernel-development's intake tells it to normalize `pallas` to `jax`."""
        text = (PACKAGE_DIR / "agents" /
                "tpu-kernel-development.md").read_text()
        # Collapse whitespace so re-wrapping the paragraph does not break this.
        self.assertIn("Record an `input_language` of `pallas` as `jax`",
                      " ".join(text.split()))


def _inline_request(**fields):
    """A request whose input arrived in the conversation, not on disk."""
    request = {
        "prompt": "Make this top-k expert selection faster on TPU7x.",
        "run_id": RUN_ID,
        "base_kernel_source":
        "import jax\n\ndef computation(x):\n    return x\n",
        "input_language": "jax",
    }
    request.update(fields)
    return request


class TestKernelRequestInlineSource(unittest.TestCase):
    """`base_kernel_source`: the input the caller cannot write to disk.

    The router holds `read` alone, so code the user pasted into the
    conversation reaches the loop only as text. Exactly one of the path and
    the text is required, and the text must say what language it is, because
    that decides the extension it is persisted under.
    """

    @classmethod
    def setUpClass(cls):
        cls.request_schema = jsonio.load_json(PACKAGE_DIR / "schemas" /
                                              "kernel-request.json")

    def _errors(self, request):
        return utils.validate_instance(request, self.request_schema)

    def test_source_with_language_validates(self):
        for language in ("jax", "pallas", "pytorch", "specification"):
            with self.subTest(language=language):
                self.assertEqual(
                    self._errors(_inline_request(input_language=language)), [])

    def test_exactly_one_of_path_and_source_with_language_is_required(self):
        refused = ("$: must match exactly one of 2 oneOf alternatives, matched"
                   " {n} (one of: base_kernel_path | base_kernel_source,"
                   " input_language)")
        neither = _request()
        del neither["base_kernel_path"]
        self.assertEqual(self._errors(neither), [refused.format(n=0)])
        unlabelled = _inline_request()
        del unlabelled["input_language"]
        self.assertEqual(self._errors(unlabelled), [refused.format(n=0)])
        both = _inline_request(base_kernel_path="work/base.py")
        self.assertEqual(self._errors(both), [refused.format(n=2)])

    def test_orchestrator_treats_pasted_text_as_a_file_it_wrote(self):
        """The specialist, which can write, puts the text on disk and then
        proceeds as for a path; the result hands back that path, and the
        content, not the request's label, decides the language."""
        text = " ".join((PACKAGE_DIR / "agents" /
                         "tpu-kernel-development.md").read_text().split())
        self.assertIn("put it on disk first**, then proceed as for a path",
                      text)
        # The "never write code" rule names this write as its exception.
        self.assertIn("The one exception is Setup step 0", text)
        self.assertIn(
            "A request carrying both `base_kernel_source` and `slot` is"
            " refused", text)
        self.assertIn(
            "`base_kernel_path` as the slot's `<kernels>` entry spells it",
            text)
        self.assertIn(
            "When step 0 wrote the input as `base_kernel.py`, leave"
            " `primary.language` `null`", text)

    def test_result_owes_the_path_with_the_slot(self):
        """Follow-ups continue a pasted kernel by path, not by re-sending it,
        so a result that resolved a slot owes the path."""
        result = jsonio.load_json(PACKAGE_DIR / "schemas" /
                                  "kernel-result.json")
        base = {"status": "needs_input", "run_id": RUN_ID, "slot": "k01"}
        self.assertEqual(utils.validate_instance(base, result), [
            "$/base_kernel_path: required property is missing when slot is"
            " present"
        ])
        self.assertEqual(
            utils.validate_instance(
                {
                    **base, "base_kernel_path":
                    "/runs/x/artifacts/tpu-kernel-development/inputs/k01/"
                    "base_kernel.py"
                }, result), [])
        # Before a slot is resolved there is no path to owe; an optimized
        # result has always resolved one (`status_requires` is the runtime's).
        self.assertEqual(
            utils.validate_instance(
                {
                    "status": "invalid_input",
                    "run_id": RUN_ID
                }, result), [])
        self.assertIn("slot", result["status_requires"]["optimized"])

    def test_router_passes_pasted_text_verbatim(self):
        """tpu-agent sends the text once, untouched, and continues by path."""
        skill = " ".join((PACKAGE_DIR / "skills" / "tpu-agent" /
                          "SKILL.md").read_text().split())
        self.assertIn("pass the text as `base_kernel_source`", skill)
        self.assertIn("Copy the text exactly as the user gave it", skill)
        self.assertIn("passes that path with the `slot`, never the text again",
                      skill)
        # The inline route is not a way around the CUDA refusal.
        self.assertIn(
            "do not write a `specification` file describing it, or"
            " send one as `base_kernel_source`", skill)
        # The instruction that produced the dead end must not come back.
        self.assertNotIn("write the description to a file and name that",
                         skill)


def _case(**fields):
    return {
        "label": "decode",
        "call_shapes": {
            "operands": ["s32[256]", "bf16[8,16,4,2,128]"],
            "outputs": ["bf16[8,16,4,2,128]", "bf16[3558,256,4,2,128]"],
        },
        **fields,
    }


class TestKernelRequestServingConfig(unittest.TestCase):
    """`serving_config` names the files the served model runs under."""

    @classmethod
    def setUpClass(cls):
        cls.request_schema = jsonio.load_json(PACKAGE_DIR / "schemas" /
                                              "kernel-request.json")

    def _errors(self, serving_config):
        return utils.validate_instance(
            _request(serving_config=serving_config,
                     serving_round={"status": "nothing_to_propose"}),
            self.request_schema)

    def test_accepts_the_onboarding_paths(self):
        self.assertEqual(
            self._errors({
                "deployments_json": "/r/inference/deployments.json",
                "launch_recipe": "/r/inference/launch_vllm.sh",
            }), [])
        self.assertEqual(
            self._errors({"deployments_json":
                          "/r/inference/deployments.json"}), [])

    def test_rejects_a_config_without_the_manifest_or_with_prose(self):
        for label, config in [
            ("no manifest", {
                "launch_recipe": "/r/launch_vllm.sh"
            }),
            ("empty path", {
                "deployments_json": ""
            }),
            ("flags as prose", {
                "deployments_json": "/r/deployments.json",
                "flags": "--max-model-len 4096"
            }),
        ]:
            with self.subTest(case=label):
                self.assertTrue(self._errors(config))

    def test_an_example_carries_every_handed_on_field(self):
        """One example shows all three artifacts a served kernel arrives with."""
        self.assertTrue(
            any({"served_calls", "serving_config", "fusion_context"} <= set(e)
                for e in self.request_schema.get("examples", [])))

    def test_orchestrator_copies_serving_config_into_state(self):
        text = " ".join((PACKAGE_DIR / "agents" /
                         "tpu-kernel-development.md").read_text().split())
        self.assertIn('"serving_config": null', text)
        self.assertIn(
            "`serving_config` is the request's `serving_config`, copied"
            " verbatim", text)

    def test_harness_inputs_name_their_sources(self):
        text = " ".join(
            (PACKAGE_DIR / "references" / "tpu-kernel-development" /
             "kernel-worker" / "generate-test-file.md").read_text().split())
        for phrase in ("the two files `state.serving_config` names",
                       "**every value in them has a source you can name**",
                       "Write `<work>/input_provenance.md`",
                       "never invent one silently"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)


class TestKernelRequestServingRound(unittest.TestCase):
    """A served kernel says what the optimization-heuristics round concluded, or why it didn't run."""

    @classmethod
    def setUpClass(cls):
        cls.request_schema = jsonio.load_json(PACKAGE_DIR / "schemas" /
                                              "kernel-request.json")

    def _errors(self, **fields):
        return utils.validate_instance(_request(**fields), self.request_schema)

    def test_a_served_kernel_without_the_round_is_refused(self):
        errors = self._errors(
            serving_config={"deployments_json": "/r/deployments.json"})
        self.assertTrue(any("serving_round" in e for e in errors), errors)

    def test_each_outcome_is_accepted(self):
        for status in ("kept_candidate", "kept_baseline", "nothing_to_propose",
                       "not_run"):
            with self.subTest(status=status):
                self.assertEqual(
                    self._errors(
                        serving_config={"deployments_json": "/r/d.json"},
                        serving_round={
                            "status": status,
                            "reason": "the user declined the round"
                        }), [])

    def test_an_unknown_outcome_or_extra_field_is_refused(self):
        for label, serving_round in [
            ("unknown status", {
                "status": "skipped"
            }),
            ("no status", {
                "reason": "kernel required"
            }),
            ("extra field", {
                "status": "not_run",
                "why": "time limit"
            }),
        ]:
            with self.subTest(case=label):
                self.assertTrue(
                    self._errors(
                        serving_config={"deployments_json": "/r/d.json"},
                        serving_round=serving_round))

    def test_a_kernel_not_served_needs_no_round(self):
        self.assertEqual(self._errors(), [])

    def test_the_orchestrator_is_told_to_pass_it(self):
        text = (PACKAGE_DIR / "skills" / "tpu-agent" /
                "SKILL.md").read_text(encoding="utf-8")
        self.assertIn("`serving_round`", text)
        self.assertIn("is not a reason", text)


class TestKernelRequestServedCalls(unittest.TestCase):
    """`served_calls` carries the calls the server makes, as typed data.

    Rewritten as prose, a 16-token decode call was read back as 16
    sequences and a 2048-token prefill call as a 2048-8192 range, and the
    loop tuned for calls the server never made.
    """

    @classmethod
    def setUpClass(cls):
        cls.request_schema = jsonio.load_json(PACKAGE_DIR / "schemas" /
                                              "kernel-request.json")

    def _errors(self, served_calls):
        return utils.validate_instance(_request(served_calls=served_calls),
                                       self.request_schema)

    def test_accepts_a_case_per_capture(self):
        served = {
            "cases": [
                _case(phase="decode",
                      per_call_us=45.26,
                      share_pct=11.2,
                      load="in/out 2048/256, concurrency 32"),
                _case(label="prefill", phase="prefill", per_call_us="1332.3"),
            ],
            "profiling_report":
            "/runs/r/artifacts/tpu-diagnostics/report.md",
        }
        self.assertEqual(self._errors(served), [])

    def test_runtime_values_are_free_form(self):
        """What a trace cannot show is keyed as the measurement named it."""
        case = _case(runtime_values={
            "live_seqs_per_call_median": 4,
            "kv_len_range": [2048, 2304],
        })
        self.assertEqual(self._errors({"cases": [case]}), [])

    def test_a_shape_keeps_its_layout(self):
        """The profiler copies the layout suffix; the boundary passes it."""
        case = _case(
            call_shapes={
                "operands": ["s32[256]{0}", "bf16[8,16,4,2,128]{4,3,2,1,0}"],
                "outputs": ["bf16[8,16,4,2,128]{4,3,2,1,0}"],
            })
        self.assertEqual(self._errors({"cases": [case]}), [])

    def test_a_case_owes_its_shapes(self):
        for missing in ("label", "call_shapes"):
            with self.subTest(missing=missing):
                case = {k: v for k, v in _case().items() if k != missing}
                self.assertTrue(self._errors({"cases": [case]}))
        shapes = {"operands": ["s32[256]"]}
        self.assertTrue(self._errors({"cases": [_case(call_shapes=shapes)]}))

    def test_rejects_malformed_cases(self):
        for label, served in [
            ("no cases", {}),
            ("unknown key on the object", {
                "cases": [_case()],
                "x": 1
            }),
            ("unknown key on a case", {
                "cases": [_case(seqs=16)]
            }),
            ("unknown phase", {
                "cases": [_case(phase="warmup")]
            }),
            ("share above 100", {
                "cases": [_case(share_pct=140)]
            }),
            ("empty shape", {
                "cases":
                [_case(call_shapes={
                    "operands": [""],
                    "outputs": []
                })]
            }),
        ]:
            with self.subTest(case=label):
                self.assertTrue(self._errors(served))

    def test_an_example_carries_served_calls(self):
        self.assertTrue(
            any("served_calls" in e
                for e in self.request_schema.get("examples", [])))

    def test_orchestrator_copies_served_calls_into_state(self):
        text = " ".join((PACKAGE_DIR / "agents" /
                         "tpu-kernel-development.md").read_text().split())
        self.assertIn('"served_calls": null', text)
        self.assertIn(
            "`served_calls` is the request's `served_calls`, copied"
            " verbatim", text)

    def test_harness_is_built_from_served_calls(self):
        text = " ".join(
            (PACKAGE_DIR / "references" / "tpu-kernel-development" /
             "kernel-worker" / "generate-test-file.md").read_text().split())
        self.assertIn("Read `state.served_calls` from `<work>/state.json`",
                      text)
        self.assertIn("**before** any other config", text)

    def test_harness_reads_the_layout_and_the_servers_readings(self):
        text = " ".join(
            (PACKAGE_DIR / "references" / "tpu-kernel-development" /
             "kernel-worker" / "generate-test-file.md").read_text().split())
        for phrase in ("`get_inputs()` cannot set the layout",
                       "copy the layout into `input_provenance.md`",
                       "keyed by its own metric names, not values per call",
                       "divided across the data-parallel replicas"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_harness_prints_a_time_per_case(self):
        """Each served case's time is visible, not only the geometric mean."""
        worker = " ".join(
            (PACKAGE_DIR / "agents" / "kernel-worker.md").read_text().split())
        template = (PACKAGE_DIR / "scripts" / "tpu-kernel-development" /
                    "test_harness_template.py").read_text()
        self.assertIn("`BASE_TIME_CASE_i`, `OPT_TIME_CASE_i`", worker)
        self.assertIn(
            'print(f"BASE_TIME_CASE_{idx}: {time_base * 1000:.6f} ms")',
            template)

    def test_no_calibration_against_device_time(self):
        """Harness wall-clock and profile device time are different clocks.

        A decode call of tens of microseconds is timed by host dispatch, so a
        ratio of the two flagged every such case; the check is withdrawn
        until it can compare device time with device time.
        """
        worker = (PACKAGE_DIR / "agents" / "kernel-worker.md").read_text()
        self.assertNotIn("harness_calibration.md", worker)


class TestKernelWorkerRequestWorkDir(unittest.TestCase):
    """Each kernel in a session works in a slot of its own."""

    @classmethod
    def setUpClass(cls):
        cls.schema = jsonio.load_json(PACKAGE_DIR / "schemas" /
                                      "kernel-worker-request.json")

    def _request(self, **fields):
        return {
            "prompt": "Run one iteration of the self-refinement loop.",
            "run_id": RUN_ID,
            **fields,
        }

    def test_accepts_a_slot(self):
        work_dir = f"/runs/{RUN_ID}/artifacts/tpu-kernel-development/k02"
        self.assertEqual(
            utils.validate_instance(self._request(work_dir=work_dir),
                                    self.schema), [])

    def test_requires_work_dir(self):
        """Without it the worker would derive a root every kernel shares."""
        self.assertTrue(utils.validate_instance(self._request(), self.schema))

    def test_rejects_anything_but_a_slot(self):
        """The shared root, the index, and a name that is not `k<NN>`."""
        for work_dir in (
                f"/runs/{RUN_ID}/artifacts/tpu-kernel-development",
                f"/runs/{RUN_ID}/artifacts/tpu-kernel-development/kernels.json",
                f"/runs/{RUN_ID}/artifacts/tpu-kernel-development/iter1",
                "/runs/not-a-run/artifacts/tpu-kernel-development/k01",
        ):
            with self.subTest(work_dir=work_dir):
                self.assertTrue(
                    utils.validate_instance(self._request(work_dir=work_dir),
                                            self.schema))

    def test_examples_validate(self):
        for example in self.schema["examples"]:
            self.assertEqual(utils.validate_instance(example, self.schema), [])

    def test_orchestrator_keys_slots_on_path_and_op(self):
        """Two kernels in one source file must not share a slot."""
        text = " ".join((PACKAGE_DIR / "agents" /
                         "tpu-kernel-development.md").read_text().split())
        self.assertIn("A kernel is the pair (`base_kernel_path`, `target_op`)",
                      text)


class TestKernelSlotRoundTrip(unittest.TestCase):
    """A caller can resume a kernel by the slot an earlier result returned."""

    @classmethod
    def setUpClass(cls):
        cls.request_schema = jsonio.load_json(PACKAGE_DIR / "schemas" /
                                              "kernel-request.json")
        cls.result_schema = jsonio.load_json(PACKAGE_DIR / "schemas" /
                                             "kernel-result.json")

    def test_result_returns_a_slot(self):
        """With the slot comes the path it is keyed on: the caller continues
        the kernel by both."""
        result = {
            "status": "invalid_input",
            "run_id": RUN_ID,
            "slot": "k02",
            "base_kernel_path": "/abs/path/model.py",
        }
        self.assertEqual(utils.validate_instance(result, self.result_schema),
                         [])

    def test_result_names_other_slots_for_the_same_file(self):
        result = {
            "status":
            "optimized",
            "run_id":
            RUN_ID,
            "slot":
            "k03",
            "base_kernel_path":
            "/abs/path/attention.py",
            "same_file_slots": [{
                "slot": "k01",
                "target_op": "ragged paged attention"
            }],
        }
        self.assertEqual(utils.validate_instance(result, self.result_schema),
                         [])

    def test_orchestrator_matches_slots_on_real_path(self):
        """One file reached through a symlink is two path strings."""
        text = " ".join((PACKAGE_DIR / "agents" /
                         "tpu-kernel-development.md").read_text().split())
        self.assertIn("**Same-file check.**", text)
        self.assertIn("the same non-null `real_path`", text)
        worker = " ".join(
            (PACKAGE_DIR / "agents" / "kernel-worker.md").read_text().split())
        self.assertIn("realpath -e <state.kernel.base_kernel_path>", worker)

    def test_router_keeps_one_spelling_per_kernel(self):
        text = " ".join((PACKAGE_DIR / "skills" / "tpu-agent" /
                         "SKILL.md").read_text().split())
        self.assertIn("in one spelling for the whole session", text)

    def test_request_takes_a_slot_back(self):
        self.assertEqual(
            utils.validate_instance(_request(slot="k02"), self.request_schema),
            [])

    def test_request_rejects_a_slot_that_is_a_path(self):
        for slot in ("../k01", "k1", "tpu-kernel-development/k01"):
            with self.subTest(slot=slot):
                self.assertTrue(
                    utils.validate_instance(_request(slot=slot),
                                            self.request_schema))


if __name__ == "__main__":
    unittest.main()


class TestKernelResultCarriesState(unittest.TestCase):
    """The fields tpu-kernel-development copies from state.json pass the result boundary.

    In the gpt-oss run no reference kernel was supplied, so
    `state.reference_trust` was `null`; copied as instructed, the result was
    rejected with `$/reference_trust: expected string, got NoneType`.
    """

    @classmethod
    def setUpClass(cls):
        cls.result_schema = jsonio.load_json(PACKAGE_DIR / "schemas" /
                                             "kernel-result.json")

    def _result(self, **fields):
        return {
            "status":
            "optimized",
            "run_id":
            RUN_ID,
            "summary":
            "Kept iteration 5: 1.1008x over gmm_v2.",
            "best_latency_ms":
            0.080312,
            "baseline_latency_ms":
            0.08841,
            "speedup_factor":
            1.1008,
            "iterations_run":
            5,
            "iterations": [{
                "iteration": 5,
                "compile_ok": True,
                "test_ok": True,
                "latency_ms": 0.080312
            }],
            "best_kernel_artifact":
            "artifacts/tpu-kernel-development/iter5/optimized.py",
            "artifacts":
            ["artifacts/tpu-kernel-development/iter5/optimized.py"],
            "input_language":
            "jax",
            **fields,
        }

    def test_a_run_without_a_reference_reports_no_trust_verdict(self):
        """`reference_trust` left out, `baseline_verified` copied as null."""
        self.assertEqual(
            utils.validate_instance(self._result(baseline_verified=None),
                                    self.result_schema), [])

    def test_a_copied_null_reference_trust_is_rejected(self):
        self.assertTrue(
            utils.validate_instance(self._result(reference_trust=None),
                                    self.result_schema))

    def test_orchestrator_leaves_out_a_null_reference_trust(self):
        text = " ".join((PACKAGE_DIR / "agents" /
                         "tpu-kernel-development.md").read_text().split())
        for phrase in ("`reference_trust` from `state.reference_trust`, left "
                       "out when that is `null`",
                       "`null` included, since it says no check was possible"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)


class TestKernelResultNoSpeedup(unittest.TestCase):
    """A kernel slower than the baseline is never kept, and a run in which no
    kernel beat the baseline is flagged rather than reported `optimized`."""

    @classmethod
    def setUpClass(cls):
        cls.result_schema = jsonio.load_json(PACKAGE_DIR / "schemas" /
                                             "kernel-result.json")

    def _errors(self, payload):
        """Both halves the SubagentStop hook applies: shape, then status."""
        return (
            utils.validate_instance(payload, self.result_schema) +
            agent_contracts.check_status_payload(self.result_schema, payload))

    def _no_speedup(self, **fields):
        return {
            "status":
            "no_speedup",
            "run_id":
            RUN_ID,
            "slot":
            "k01",
            "base_kernel_path":
            "/abs/path/decode_gqa_attention.py",
            "best_kernel_artifact":
            "artifacts/tpu-kernel-development/k01/iter1/optimized.py",
            "baseline_latency_ms":
            0.0439767,
            "best_latency_ms":
            0.047999,
            "speedup_factor":
            0.9162,
            "iterations_run":
            3,
            "iterations": [{
                "iteration": 1,
                "compile_ok": True,
                "test_ok": True,
                "latency_ms": 0.047999,
                "speedup": 0.9162
            }, {
                "iteration": 2,
                "compile_ok": True,
                "test_ok": True,
                "latency_ms": 0.050278,
                "speedup": 0.9289
            }, {
                "iteration": 3,
                "compile_ok": True,
                "test_ok": True,
                "latency_ms": 0.049665,
                "speedup": 0.9262
            }],
            **fields,
        }

    def test_accepts_no_speedup_with_its_measurements(self):
        """Three passing kernels, all slower than base.py."""
        self.assertEqual(self._errors(self._no_speedup()), [])

    def test_no_speedup_owes_its_measurements(self):
        payload = self._no_speedup()
        for field in ("best_latency_ms", "speedup_factor", "iterations"):
            del payload[field]
        errors = self._errors(payload)
        self.assertEqual(len(errors), 1)
        for field in ("best_latency_ms", "speedup_factor", "iterations"):
            with self.subTest(field=field):
                self.assertIn(field, errors[0])

    def test_worker_keeps_only_a_kernel_that_beat_the_baseline(self):
        text = " ".join(
            (PACKAGE_DIR / "agents" / "kernel-worker.md").read_text().split())
        self.assertIn(
            "If `compile_ok` is True, `test_ok` is True, `speedup` is above"
            " 1.0, and `optimized_time` is better (smaller) than"
            " `state.best_optimized_time`", text)

    def test_worker_records_the_tuned_kernels_speedup(self):
        """Phase 3 timed the untuned kernel; the gate needs the one on disk."""
        text = " ".join(
            (PACKAGE_DIR / "agents" / "kernel-worker.md").read_text().split())
        self.assertIn(
            "it is the `SPEEDUP` line in the `stdout` of the `all_results`"
            " entry whose `config` is `best_config`", text)

    def test_tuned_trials_print_a_paired_speedup(self):
        """The line the worker reads exists in every trial's stdout."""
        harness = (PACKAGE_DIR / "scripts" / "tpu-kernel-development" /
                   "test_harness_template.py").read_text()
        self.assertIn('print(f"SPEEDUP: {geo_mean_speedup:.4f}")', harness)
        autotune = (PACKAGE_DIR / "scripts" / "tpu-kernel-development" /
                    "run_autotune.py").read_text()
        self.assertIn('"stdout": stdout,', autotune)

    def test_orchestrator_returns_no_speedup(self):
        text = " ".join((PACKAGE_DIR / "agents" /
                         "tpu-kernel-development.md").read_text().split())
        for phrase in ("**no kernel this run built beat the baseline**",
                       "kernels compiled and passed, but none beat the"
                       " baseline (step 4) -> `no_speedup`",
                       "Never report such a run as `optimized`",
                       "its `speedup` is not above 1.0, or its"
                       " `optimized_time` is not below"):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, text)

    def test_router_integrates_only_an_optimized_kernel(self):
        skill = " ".join((PACKAGE_DIR / "skills" / "tpu-agent" /
                          "SKILL.md").read_text().split())
        self.assertIn(
            "Integrate a kernel only when tpu-kernel-development returns"
            " `optimized`", skill)
