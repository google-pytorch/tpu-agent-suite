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
"""Unit tests for `compile_verdict.py`."""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import compile_verdict  # noqa: E402


def _encode_varint(val: int) -> bytes:
    out = bytearray()
    while True:
        chunk = val & 0x7F
        val >>= 7
        if val:
            out.append(chunk | 0x80)
        else:
            out.append(chunk)
            return bytes(out)


def _encode_bytes_field(field_number: int, payload: bytes) -> bytes:
    tag = (field_number << 3) | 2
    return _encode_varint(tag) + _encode_varint(len(payload)) + payload


def _encode_varint_field(field_number: int, val: int) -> bytes:
    tag = (field_number << 3) | 0
    return _encode_varint(tag) + _encode_varint(val)


def _build_hlo_proto(opcodes: list[str]) -> bytes:
    instructions = b"".join(
        _encode_bytes_field(2, _encode_bytes_field(2, op.encode("utf-8")))
        for op in opcodes)
    computation = instructions
    return _encode_bytes_field(3, computation)


def _build_memory_report(hbm_bytes: int, host_bytes: int = 0) -> bytes:
    hbm_entry = _encode_varint_field(1, hbm_bytes) + _encode_varint_field(
        2, compile_verdict.COLOR_HBM)
    out = _encode_bytes_field(1, hbm_entry)
    if host_bytes:
        host_entry = _encode_varint_field(1,
                                          host_bytes) + _encode_varint_field(
                                              2, compile_verdict.COLOR_HOST)
        out += _encode_bytes_field(1, host_entry)
    return out


class CompileVerdictTest(unittest.TestCase):
    """Tests pure-stdlib HLO wire parsing, error classification, and compile verdicts."""

    def test_hlo_stats_counts_async_start_once_and_skips_done(self):
        proto = _build_hlo_proto([
            "parameter",
            "all-gather-start",
            "all-gather-done",
            "dot",
            "reduce-scatter",
            "all-to-all-start",
            "all-to-all-done",
        ])
        stats = compile_verdict.hlo_stats(proto)
        self.assertEqual(stats.instruction_count, 7)
        self.assertEqual(stats.collectives["all-gather"], 1)
        self.assertEqual(stats.collectives["reduce-scatter"], 1)
        self.assertEqual(stats.collectives["all-to-all"], 1)
        self.assertEqual(stats.collectives["all-reduce"], 0)

    def test_parse_peak_memory_extracts_hbm_and_host_colors(self):
        report = _build_memory_report(hbm_bytes=16 * (1 << 30),
                                      host_bytes=2 * (1 << 30))
        peaks = compile_verdict.parse_peak_memory(report)
        self.assertEqual(peaks[compile_verdict.COLOR_HBM], 16 * (1 << 30))
        self.assertEqual(peaks[compile_verdict.COLOR_HOST], 2 * (1 << 30))

    def test_classify_compile_error_precedence(self):
        self.assertEqual(
            compile_verdict.classify_compile_error(
                "CompileTimeScopedVmemOom: ran out of memory in memory space vmem"
            ),
            compile_verdict.ERROR_VMEM_OOM,
        )
        self.assertEqual(
            compile_verdict.classify_compile_error(
                "RESOURCE_EXHAUSTED: CompileTimeHbmOom exceeds available HBM"),
            compile_verdict.ERROR_HBM_OOM,
        )
        self.assertEqual(
            compile_verdict.classify_compile_error(
                "ValueError: create_device_mesh() failed; set allow_split_physical_axes=True"
            ),
            compile_verdict.ERROR_MESH_SPLIT,
        )

    def test_decide_verdict_landed_noop_and_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            base_dir = pathlib.Path(tmp) / "base"
            same_dir = pathlib.Path(tmp) / "same"
            diff_dir = pathlib.Path(tmp) / "diff"
            empty_dir = pathlib.Path(tmp) / "empty"
            for d in (base_dir, same_dir, diff_dir, empty_dir):
                d.mkdir()

            base_hlo = _build_hlo_proto(
                ["parameter", "all-gather", "all-gather", "dot"])
            diff_hlo = _build_hlo_proto(
                ["parameter", "all-gather", "all-reduce", "dot"])
            base_mem = _build_memory_report(20 * (1 << 30))
            diff_mem = _build_memory_report(14 * (1 << 30))

            (base_dir / "module_0001.jit_train_step.after_optimizations.hlo.pb"
             ).write_bytes(base_hlo)
            (base_dir /
             "module_0001.jit_train_step.after_optimizations-memory-usage-report.pb"
             ).write_bytes(base_mem)

            (same_dir / "module_0001.jit_train_step.after_optimizations.hlo.pb"
             ).write_bytes(base_hlo)
            (same_dir /
             "module_0001.jit_train_step.after_optimizations-memory-usage-report.pb"
             ).write_bytes(base_mem)

            (diff_dir / "module_0001.jit_train_step.after_optimizations.hlo.pb"
             ).write_bytes(diff_hlo)
            (diff_dir /
             "module_0001.jit_train_step.after_optimizations-memory-usage-report.pb"
             ).write_bytes(diff_mem)

            landed = compile_verdict.decide_verdict(
                compile_passed=True,
                compile_log="",
                baseline_dir=str(base_dir),
                candidate_dir=str(diff_dir),
                module=None,
            )
            self.assertEqual(landed.verdict, compile_verdict.Verdict.LANDED)

            noop = compile_verdict.decide_verdict(
                compile_passed=True,
                compile_log="",
                baseline_dir=str(base_dir),
                candidate_dir=str(same_dir),
                module=None,
            )
            self.assertEqual(noop.verdict, compile_verdict.Verdict.NO_OP)

            dropped = compile_verdict.decide_verdict(
                compile_passed=False,
                compile_log="RESOURCE_EXHAUSTED: out of memory on hbm",
                baseline_dir=str(base_dir),
                candidate_dir=str(empty_dir),
                module=None,
            )
            self.assertEqual(dropped.verdict, compile_verdict.Verdict.DROPPED)
            self.assertEqual(dropped.error_class,
                             compile_verdict.ERROR_HBM_OOM)

    def test_unknown_config_paths_detection(self):
        plan_paths = compile_verdict.plan_paths_from_json(
            '{"trainer": {"ici_mesh": {"fsdp": 8, "tensor": 4}, "remat": "full"}}'
        )
        unknown = compile_verdict.find_unknown_paths(
            {
                "trainer": {
                    "ici_mesh": {
                        "fsdp": 4,
                        "tensor": 8
                    },
                    "typo_knob": True
                }
            },
            plan_paths,
        )
        self.assertEqual(unknown, ["trainer.typo_knob"])


if __name__ == "__main__":
    unittest.main()
