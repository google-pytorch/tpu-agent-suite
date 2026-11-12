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
"""Ahead-of-time (AOT) compile verdict and raw XLA HLO dump verification.

A candidate's AOT compilation exiting 0 does not guarantee the sharding change
landed: trainers can swallow trace-time errors and exit 0 without an HLO dump,
or config loaders can silently write overrides at paths the model config does
not read. This module decides, from local artifacts alone (the compile log, XLA
dumps from `--xla_dump_to`, and an optional saved baseline config plan), whether
a candidate:

* `LANDED`: compiled and changed the program (sharding, mesh devices, or the
  optimized HLO's instruction / collective counts or peak HBM);
* `NO_OP`: compiled to the exact same program as the baseline;
* `DROPPED`: did not compile, produced no optimized HLO, or wrote config paths
  absent from the baseline plan.

Per-module evidence (peak HBM per memory color, collective opcode counts, total
instruction count) is extracted via pure-stdlib protobuf wire parsing, with zero
external protobuf or XLA dependencies.
"""

from __future__ import annotations

import dataclasses
import enum
import json
import os
import re
import subprocess
from collections.abc import (Callable, Collection, Iterable, Iterator, Mapping,
                             Sequence)
from typing import Any

# Error classes for DROPPED candidates.
ERROR_HBM_OOM = "hbm_oom"
ERROR_VMEM_OOM = "vmem_oom"
ERROR_SHARDING = "sharding_error"
ERROR_MESH_SPLIT = "mesh_split_required"
ERROR_PREFETCH_RET_CHECK = "explicit_prefetch_ret_check"
ERROR_TILING = "tiling_failure"
ERROR_FLAG = "unknown_flag"
ERROR_UNKNOWN_CONFIG_PATH = "unknown_config_path"
ERROR_NO_HLO = "no_hlo_dump"
ERROR_TIMEOUT = "timeout"
ERROR_COMPILE = "compile_error"

# XLA memory-space colors in `*-memory-usage-report.pb`.
COLOR_HBM = 0
COLOR_HOST = 5

COLLECTIVE_OPCODES = (
    "all-gather",
    "all-reduce",
    "all-to-all",
    "ragged-all-to-all",
    "reduce-scatter",
    "collective-permute",
)

_OPTIMIZED_HLO_SUFFIX = ".after_optimizations.hlo.pb"
_MEMORY_REPORT_SUFFIX = ".after_optimizations-memory-usage-report.pb"
_BEFORE_OPT_SUFFIX = ".before_optimizations.hlo.pb"
_MODULE_NUMBER_RE = re.compile(r"module_(\d+)\.")

# Ordered: the first matching class wins. Flag errors abort before anything
# compiles; VMEM is checked before HBM because VMEM OOM messages also contain
# "ran out of memory".
_ERROR_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        ERROR_FLAG,
        re.compile(r"Unknown command line flag|Flags parsing error",
                   re.IGNORECASE),
    ),
    (
        ERROR_VMEM_OOM,
        re.compile(
            r"CompileTimeScopedVmemOom|out of memory in memory space vmem",
            re.IGNORECASE,
        ),
    ),
    (
        ERROR_HBM_OOM,
        re.compile(
            r"CompileTimeHbmOom|memory space hbm|out of memory on hbm"
            r"|exceeds available HBM|HLO temporaries|exceeded hbm capacity",
            re.IGNORECASE,
        ),
    ),
    (
        ERROR_MESH_SPLIT,
        re.compile(
            r"create_device_mesh\(.*\)`?\s*failed"
            r"|set\s+`?allow_split_physical_axes"
            r"|Failed to find assignment for logical_axis_index",
            re.IGNORECASE,
        ),
    ),
    (
        ERROR_SHARDING,
        re.compile(
            r"ShardingTypeError|sharding_lowering|not divisible|must be"
            r" divisible",
            re.IGNORECASE,
        ),
    ),
    (
        ERROR_PREFETCH_RET_CHECK,
        re.compile(r"RET_CHECK[^\n]*prefetch|prefetch[^\n]*RET_CHECK",
                   re.IGNORECASE),
    ),
    (
        ERROR_TILING,
        re.compile(
            r"Failed to find a tiling|no valid tiling|tiling failed",
            re.IGNORECASE,
        ),
    ),
    (
        ERROR_HBM_OOM,
        re.compile(
            r"RESOURCE_EXHAUSTED|ResourceExhausted|out[ _]of[ _]memory"
            r"|allocation exceeds memory|hbm oom",
            re.IGNORECASE,
        ),
    ),
)

_FATAL_LINE_RE = re.compile(
    r"^.*(?:Top-level exception:|Error: |Error\(|FATAL ).*$", re.MULTILINE)

_FREE_FORM_PATH_RE = re.compile(
    r"(^|\.)(interpreter_args|extra_server_args|extra_client_args)\.(-{0,2})(xla_|partir_)"
)
_LIST_INDEX_RE = re.compile(r"\[([^\]]+)\]")
_MESH_DICT_PARENT_RE = re.compile(r"(^|\.)\w*mesh$")
MODEL_CONFIG_KNOB_FIELDS: frozenset[str] = frozenset(
    {"logits_calculation_chunk_mode"})

# Offline simulator label when simulated step time is within tolerance of
# baseline while memory placement (host offload / peak HBM) changed.
SIM_INCONCLUSIVE = "SIM_INCONCLUSIVE: needs real-TPU A/B"
SIM_INCONCLUSIVE_STEP_TOLERANCE = 0.03
SIM_INCONCLUSIVE_HBM_DELTA_BYTES = 1 << 30
_HOST_PLACEMENT_KEY_RE = re.compile(
    r"(^|\.)(store_\w*_on_host|\w*offload\w*|manually_store_block_residual)$",
    re.IGNORECASE,
)


class Verdict(enum.Enum):
    """Whether a candidate's change reached the compiled program."""

    LANDED = "LANDED"
    NO_OP = "NO_OP"
    DROPPED = "DROPPED"


@dataclasses.dataclass(frozen=True)
class ModuleFiles:
    """Dump files of one HLO module, or None when absent."""

    optimized_hlo: str | None = None
    memory_report: str | None = None
    before_optimizations: str | None = None
    module: str | None = None


@dataclasses.dataclass(frozen=True)
class HloStats:
    """Op statistics of an optimized HLO module."""

    instruction_count: int = 0
    collectives: Mapping[str, int] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(frozen=True)
class ModuleEvidence:
    """Evidence extracted from one trial's dump for the compared module."""

    module: str | None = None
    files: ModuleFiles = ModuleFiles()
    stats: HloStats | None = None
    peak_memory: Mapping[int, int] = dataclasses.field(default_factory=dict)

    @property
    def peak_hbm_bytes(self) -> int | None:
        """Peak HBM bytes of the module, or None if unknown."""
        return self.peak_memory.get(COLOR_HBM)

    @property
    def peak_host_bytes(self) -> int | None:
        """Peak host memory bytes of the module, or None if unknown."""
        return self.peak_memory.get(COLOR_HOST)


@dataclasses.dataclass(frozen=True)
class VerdictResult:
    """Verdict of one candidate with its evidence."""

    verdict: Verdict
    error_class: str | None = None
    evidence: str = ""
    module_evidence: ModuleEvidence | None = None
    sharding_check: Mapping[str, Any] | None = None


class UnknownConfigPathError(ValueError):
    """A candidate's config trial writes paths absent from the baseline plan."""

    def __init__(self, paths: Sequence[str]):
        self.paths = list(paths)
        super().__init__(
            f"{len(self.paths)} config trial path(s) not in the baseline plan: "
            + ", ".join(self.paths))


# ---------------------------------------------------------------------------
# Compile error classification.
# ---------------------------------------------------------------------------


def _match_error_class(text: str) -> str:
    for error_class, pattern in _ERROR_PATTERNS:
        if pattern.search(text):
            return error_class
    return ERROR_COMPILE


def classify_compile_error(log: str) -> str:
    """Returns the error class of a failed compile from its log or exception."""
    log = log or ""
    fatal_lines = _FATAL_LINE_RE.findall(log)
    if fatal_lines:
        error_class = _match_error_class(fatal_lines[-1])
        if error_class != ERROR_COMPILE:
            return error_class
    return _match_error_class(log)


def error_context(log: str, max_lines: int = 3) -> str:
    """Returns the last fatal-looking lines of a log."""
    return "\n".join(
        line.strip()[:1000]
        for line in _FATAL_LINE_RE.findall(log or "")[-max_lines:])


# ---------------------------------------------------------------------------
# Raw protobuf wire parsing.
# ---------------------------------------------------------------------------


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        if i >= len(buf):
            raise ValueError("Truncated varint.")
        byte = buf[i]
        i += 1
        result |= (byte & 0x7F) << shift
        shift += 7
        if not byte & 0x80:
            return result, i


def _fields(buf: bytes,
            start: int = 0,
            end: int | None = None) -> Iterator[tuple[int, int, Any]]:
    """Yields `(field_number, wire_type, value)` of a serialized protobuf message."""
    end = len(buf) if end is None else end
    i = start
    while i < end:
        key, i = _varint(buf, i)
        field_number, wire_type = key >> 3, key & 7
        if wire_type == 0:
            value, i = _varint(buf, i)
            yield field_number, wire_type, value
        elif wire_type == 2:
            length, i = _varint(buf, i)
            if i + length > end:
                raise ValueError("Truncated length-delimited field.")
            yield field_number, wire_type, (i, i + length)
            i += length
        elif wire_type == 1:
            i += 8
        elif wire_type == 5:
            i += 4
        else:
            raise ValueError(f"Unsupported wire type {wire_type}.")
    if i != end:
        raise ValueError("Message overruns its bounds.")


def _module_range(buf: bytes) -> tuple[int, int]:
    """Returns the `HloModuleProto` byte range in a dumped `*.hlo.pb`."""
    for field_number, wire_type, value in _fields(buf):
        if field_number == 1 and wire_type == 2:
            try:
                if any(fn == 3 and wt == 2
                       for fn, wt, _ in _fields(buf, *value)):
                    return value
            except ValueError:
                pass
            break
    return (0, len(buf))


def hlo_stats(hlo_proto: bytes) -> HloStats:
    """Counts instructions and collective opcodes of a serialized HLO module.

    Async collectives are counted once (`*-start`); `*-done` is skipped.
    """
    module = _module_range(hlo_proto)
    instructions = 0
    collectives = {op: 0 for op in COLLECTIVE_OPCODES}
    for fn, wt, computation in _fields(hlo_proto, *module):
        if fn != 3 or wt != 2:
            continue
        for fn2, wt2, instruction in _fields(hlo_proto, *computation):
            if fn2 != 2 or wt2 != 2:
                continue
            instructions += 1
            for fn3, wt3, value in _fields(hlo_proto, *instruction):
                if fn3 == 2 and wt3 == 2:
                    opcode = hlo_proto[value[0]:value[1]].decode(
                        errors="replace")
                    if opcode.endswith("-done"):
                        break
                    base = opcode.removesuffix("-start")
                    if base in collectives:
                        collectives[base] += 1
                    break
    return HloStats(instruction_count=instructions, collectives=collectives)


def parse_peak_memory(report: bytes) -> dict[int, int]:
    """Returns peak bytes per memory color from `*-memory-usage-report.pb`."""
    peaks: dict[int, int] = {}
    for _, wire_type, value in _fields(report):
        if wire_type != 2:
            continue
        entry: dict[int, int] = {}
        try:
            for fn2, wt2, v2 in _fields(report, *value):
                if wt2 == 0:
                    entry[fn2] = v2
        except ValueError:
            continue
        if 1 not in entry:
            continue
        color = entry.get(2, 0)
        peaks[color] = max(peaks.get(color, 0), entry[1])
    return peaks


# ---------------------------------------------------------------------------
# Dump discovery.
# ---------------------------------------------------------------------------


def _list_dump_files(dump_dir: str) -> list[str]:
    """Lists files in `dump_dir` and its direct subdirectories."""
    try:
        if not dump_dir or not os.path.exists(dump_dir):
            return []
        paths: list[str] = []
        for entry in sorted(os.listdir(dump_dir)):
            name = str(entry).rstrip("/")
            path = os.path.join(dump_dir, name)
            if os.path.isdir(path):
                paths.extend(
                    os.path.join(path,
                                 str(f).rstrip("/"))
                    for f in sorted(os.listdir(path)))
            else:
                paths.append(path)
        return paths
    except OSError:
        return []


def has_optimized_hlo(dump_dir: str) -> bool:
    """Whether `dump_dir` holds at least one optimized HLO module."""
    return any(
        p.endswith(_OPTIMIZED_HLO_SUFFIX) for p in _list_dump_files(dump_dir))


def _latest(paths: Iterable[str]) -> str | None:
    """Returns the path with the highest XLA module number."""

    def key(path: str) -> tuple[int, str]:
        match = _MODULE_NUMBER_RE.search(os.path.basename(path))
        return (int(match.group(1)) if match else -1, path)

    ordered = sorted(paths, key=key)
    return ordered[-1] if ordered else None


_DUMPED_MODULE_RE = re.compile(r"^module_\d+\.([^.]+)\.")
MODULE_NOT_FOUND_NOTE = "not found in the dump"


def resolve_module_name(dump_dir: str, module: str) -> str | None:
    """Returns the dumped module name matching `module` (exact first, else shortest prefix)."""
    names: set[str] = set()
    for path in _list_dump_files(dump_dir):
        match = _DUMPED_MODULE_RE.match(os.path.basename(path))
        if match:
            names.add(match.group(1))
    if module in names:
        return module
    prefixed = sorted((n for n in names if n.startswith(module)), key=len)
    return prefixed[0] if prefixed else None


def find_module_files(dump_dir: str, module: str) -> ModuleFiles:
    """Finds the dump files of HLO module `module` (e.g. `jit_train_step`)."""
    resolved = resolve_module_name(dump_dir, module) or module
    marker = f".{resolved}."
    files = [
        p for p in _list_dump_files(dump_dir) if marker in os.path.basename(p)
    ]
    return ModuleFiles(
        optimized_hlo=_latest(p for p in files
                              if p.endswith(_OPTIMIZED_HLO_SUFFIX)),
        memory_report=_latest(p for p in files
                              if p.endswith(_MEMORY_REPORT_SUFFIX)),
        before_optimizations=_latest(p for p in files
                                     if p.endswith(_BEFORE_OPT_SUFFIX)),
        module=resolved if files else None,
    )


def _read_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def collect_module_evidence(dump_dir: str,
                            module: str | None) -> ModuleEvidence:
    """Extracts optimized-HLO stats and peak memory of `module` in `dump_dir`."""
    if not module:
        # When no module name is specified, auto-pick the latest optimized HLO
        # module in `dump_dir` if one exists.
        candidates = [
            p for p in _list_dump_files(dump_dir)
            if p.endswith(_OPTIMIZED_HLO_SUFFIX)
        ]
        latest = _latest(candidates)
        if not latest:
            return ModuleEvidence()
        match = _DUMPED_MODULE_RE.match(os.path.basename(latest))
        if not match:
            return ModuleEvidence()
        module = match.group(1)
    files = find_module_files(dump_dir, module)
    stats = None
    peak_memory: dict[int, int] = {}
    if files.optimized_hlo:
        try:
            stats = hlo_stats(_read_bytes(files.optimized_hlo))
        except ValueError:
            stats = None
    if files.memory_report:
        try:
            peak_memory = parse_peak_memory(_read_bytes(files.memory_report))
        except ValueError:
            peak_memory = {}
    return ModuleEvidence(
        module=files.module or module,
        files=files,
        stats=stats,
        peak_memory=peak_memory,
    )


# ---------------------------------------------------------------------------
# Optional external sharding checker subprocess support.
# ---------------------------------------------------------------------------

Runner = Callable[[Sequence[str]], tuple[int, str]]


def _subprocess_runner(cmd: Sequence[str]) -> tuple[int, str]:
    proc = subprocess.run(
        list(cmd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout


def parse_sharding_check_output(output: str) -> dict[str, Any]:
    """Parses `key=value` stdout of an external HLO sharding diff checker."""
    result: dict[str, Any] = {}
    for line in output.splitlines():
        key, sep, value = line.strip().partition("=")
        if not sep:
            continue
        if key == "has_changes":
            result[key] = value.strip() == "True"
        elif key in ("changed_count", "added_count", "removed_count"):
            try:
                result[key] = int(value)
            except ValueError:
                continue
        elif key in ("mesh_devices_before", "mesh_devices_after"):
            result[key] = value.strip()
    match = re.search(r"distinct_shardings_diff: added=(\d+) removed=(\d+)",
                      output)
    if match:
        result["distinct_added"] = int(match.group(1))
        result["distinct_removed"] = int(match.group(2))
    return result


def run_sharding_check(
    checker_cmd: Sequence[str],
    baseline_dir: str,
    candidate_dir: str,
    module: str | None,
    runner: Runner = _subprocess_runner,
) -> dict[str, Any]:
    """Runs an external HLO sharding diff binary on two dump directories."""
    cmd = [
        *checker_cmd,
        f"--baseline_dir={baseline_dir}",
        f"--candidate_dir={candidate_dir}",
    ]
    if module:
        cmd.append(f"--computation={module}")
    try:
        returncode, output = runner(cmd)
    except OSError as e:
        return {"error": f"could not run sharding check: {e}"}
    result = parse_sharding_check_output(output)
    if returncode or "has_changes" not in result:
        return {"error": f"exit {returncode}: {output[-500:].strip()}"}
    return result


# ---------------------------------------------------------------------------
# Verdict.
# ---------------------------------------------------------------------------


def _same_file_bytes(path_a: str | None, path_b: str | None) -> bool:
    if not path_a or not path_b:
        return False
    return _read_bytes(path_a) == _read_bytes(path_b)


def _stats_changes(baseline: ModuleEvidence,
                   candidate: ModuleEvidence) -> list[str]:
    """Describes optimized-HLO stat differences between candidate and baseline."""
    changes: list[str] = []
    if baseline.stats and candidate.stats:
        if baseline.stats.instruction_count != candidate.stats.instruction_count:
            changes.append(f"instructions {baseline.stats.instruction_count}"
                           f"->{candidate.stats.instruction_count}")
        for op in COLLECTIVE_OPCODES:
            before = baseline.stats.collectives.get(op, 0)
            after = candidate.stats.collectives.get(op, 0)
            if before != after:
                changes.append(f"{op} {before}->{after}")
    if (baseline.peak_hbm_bytes is not None
            and candidate.peak_hbm_bytes is not None
            and baseline.peak_hbm_bytes != candidate.peak_hbm_bytes):
        changes.append(f"peak HBM {format_bytes(baseline.peak_hbm_bytes)}"
                       f"->{format_bytes(candidate.peak_hbm_bytes)}")
    return changes


def _sharding_evidence(check: Mapping[str, Any]) -> str:
    parts = [f"sharding changed={check.get('changed_count', 0)}"]
    if check.get("added_count") or check.get("removed_count"):
        parts.append(f"added={check.get('added_count', 0)}"
                     f" removed={check.get('removed_count', 0)}")
    before = check.get("mesh_devices_before")
    after = check.get("mesh_devices_after")
    if before != after:
        parts.append(f"mesh devices {before}->{after}")
    return ", ".join(parts)


def decide_verdict(
    *,
    compile_passed: bool,
    compile_log: str,
    baseline_dir: str | None,
    candidate_dir: str,
    module: str | None,
    checker_cmd: (Sequence[str] | Callable[[], Sequence[str] | None]
                  | None) = None,
    runner: Runner = _subprocess_runner,
) -> VerdictResult:
    """Decides `LANDED`, `NO_OP`, or `DROPPED` for one candidate."""
    if not compile_passed:
        return VerdictResult(
            Verdict.DROPPED,
            error_class=classify_compile_error(compile_log),
            evidence="compile failed",
        )
    if not has_optimized_hlo(candidate_dir):
        error_class = classify_compile_error(compile_log)
        if error_class == ERROR_COMPILE:
            error_class = ERROR_NO_HLO
        return VerdictResult(
            Verdict.DROPPED,
            error_class=error_class,
            evidence="compile exited 0 but produced no optimized HLO dump",
        )

    candidate = collect_module_evidence(candidate_dir, module)
    if not baseline_dir or not has_optimized_hlo(baseline_dir):
        return VerdictResult(
            Verdict.LANDED,
            evidence=
            "compiled; no baseline dump to compare against (unverified)",
            module_evidence=candidate,
        )
    baseline = collect_module_evidence(baseline_dir, module)
    if candidate.files.optimized_hlo is None:
        if baseline.files.optimized_hlo is not None:
            error_class = classify_compile_error(compile_log)
            if error_class == ERROR_COMPILE:
                error_class = ERROR_NO_HLO
            return VerdictResult(
                Verdict.DROPPED,
                error_class=error_class,
                evidence=f"no optimized HLO for module {module!r}",
                module_evidence=candidate,
            )
        return VerdictResult(
            Verdict.LANDED,
            evidence=(f"compiled; module {module!r} {MODULE_NOT_FOUND_NOTE}"
                      " (unverified)"),
            module_evidence=candidate,
        )
    if _same_file_bytes(baseline.files.optimized_hlo,
                        candidate.files.optimized_hlo):
        return VerdictResult(
            Verdict.NO_OP,
            evidence="optimized HLO byte-identical to baseline",
            module_evidence=candidate,
        )

    check: dict[str, Any] | None = None
    resolved_cmd = checker_cmd() if callable(checker_cmd) else checker_cmd
    if resolved_cmd:
        check = run_sharding_check(
            resolved_cmd,
            baseline_dir,
            candidate_dir,
            candidate.module or module,
            runner=runner,
        )
    stat_changes = _stats_changes(baseline, candidate)
    if check and "has_changes" in check:
        if check["has_changes"]:
            evidence = _sharding_evidence(check)
            if stat_changes:
                evidence += "; " + ", ".join(stat_changes)
            return VerdictResult(
                Verdict.LANDED,
                evidence=evidence,
                module_evidence=candidate,
                sharding_check=check,
            )
        if stat_changes:
            return VerdictResult(
                Verdict.LANDED,
                evidence="sharding unchanged; " + ", ".join(stat_changes),
                module_evidence=candidate,
                sharding_check=check,
            )
        return VerdictResult(
            Verdict.NO_OP,
            evidence=(
                "sharding, mesh devices, op counts and peak HBM equal baseline"
            ),
            module_evidence=candidate,
            sharding_check=check,
        )

    note = (f"sharding check unavailable ({check['error']})"
            if check and "error" in check else "sharding check skipped")
    if stat_changes:
        return VerdictResult(
            Verdict.LANDED,
            evidence=f"{note}; " + ", ".join(stat_changes),
            module_evidence=candidate,
            sharding_check=check,
        )
    return VerdictResult(
        Verdict.LANDED,
        evidence=(
            f"{note}; (unverified; op counts and peak HBM equal baseline)"),
        module_evidence=candidate,
        sharding_check=check,
    )


# ---------------------------------------------------------------------------
# Config plan override path check.
# ---------------------------------------------------------------------------


def _is_tuple_marker(value: Any) -> bool:
    return isinstance(value, Mapping) and value.get("__type__") == "tuple"


def flatten_trial_paths(trial: Mapping[str, Any],
                        prefix: str = "") -> list[str]:
    """Returns the leaf paths written for a nested config override dict."""
    paths: list[str] = []
    for key, value in trial.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value,
                      Mapping) and value and not _is_tuple_marker(value):
            paths.extend(flatten_trial_paths(value, path))
        else:
            paths.append(path)
    return paths


def _plan_nodes_from_json(value: Any, path: str, nodes: set[str]) -> None:
    if path:
        nodes.add(path)
    if _is_tuple_marker(value):
        return
    if isinstance(value, Mapping):
        for key, child in value.items():
            _plan_nodes_from_json(child, f"{path}.{key}" if path else str(key),
                                  nodes)
    elif isinstance(value, list):
        for i, child in enumerate(value):
            _plan_nodes_from_json(child, f"{path}.{i}", nodes)


def _add_with_ancestors(path: str, nodes: set[str]) -> None:
    parts = path.split(".")
    for i in range(1, len(parts) + 1):
        nodes.add(".".join(parts[:i]))


def plan_paths_from_json(plan_json: str) -> frozenset[str]:
    """Returns every node path (leaves and parents) of a saved config plan JSON."""
    nodes: set[str] = set()
    _plan_nodes_from_json(json.loads(plan_json), "", nodes)
    return frozenset(nodes)


def plan_paths_from_flattened(flattened: str) -> frozenset[str]:
    """Returns every node path of a `*_flattened.txt` config plan."""
    nodes: set[str] = set()
    for line in flattened.splitlines():
        path, sep, _ = line.partition(" : ")
        path = path.strip()
        if not sep or not path:
            continue
        _add_with_ancestors(_LIST_INDEX_RE.sub(r".\1", path), nodes)
    return frozenset(nodes)


def load_plan_paths(save_config_path: str) -> frozenset[str] | None:
    """Loads baseline config plan paths from JSON or its `*_flattened.txt` sibling."""
    if save_config_path and os.path.exists(save_config_path):
        try:
            with open(save_config_path, encoding="utf-8") as f:
                return plan_paths_from_json(f.read())
        except (ValueError, OSError):
            pass
    flattened_path = f"{os.path.splitext(save_config_path)[0]}_flattened.txt"
    if save_config_path and os.path.exists(flattened_path):
        with open(flattened_path, encoding="utf-8") as f:
            paths = plan_paths_from_flattened(f.read())
        return paths or None
    return None


def find_unknown_paths(trial: Mapping[str, Any],
                       plan_paths: Collection[str]) -> list[str]:
    """Returns config override paths absent from `plan_paths`."""
    known = (plan_paths if isinstance(plan_paths,
                                      (set, frozenset)) else set(plan_paths))
    unknown: list[str] = []
    for path in flatten_trial_paths(trial):
        if path in known or _FREE_FORM_PATH_RE.search(path):
            continue
        parent, _, leaf = path.rpartition(".")
        if parent in known and _MESH_DICT_PARENT_RE.search(parent):
            continue
        if leaf in MODEL_CONFIG_KNOB_FIELDS and parent in known:
            continue
        unknown.append(path)
    return sorted(unknown)


def changes_memory_placement(
    sharding_overrides: Mapping[str, Any] | None,
    remat: Any = None,
) -> bool:
    """True iff a candidate changes host offload / memory placement knobs."""
    for key, value in (sharding_overrides or {}).items():
        key_str = str(key).strip()
        if _HOST_PLACEMENT_KEY_RE.search(key_str):
            return True
        if (key_str.rsplit(".", 1)[-1] == "remat"
                and "OFFLOAD" in str(value).upper()):
            return True
    return "OFFLOAD" in str(remat).upper()


def sim_inconclusive_reason(
    candidate: Mapping[str, Any],
    baseline: Mapping[str, Any] | None,
    step_tolerance: float = SIM_INCONCLUSIVE_STEP_TOLERANCE,
    hbm_delta_bytes: int = SIM_INCONCLUSIVE_HBM_DELTA_BYTES,
) -> str | None:
    """Returns a `SIM_INCONCLUSIVE` note when an offline simulator cannot rank a candidate."""
    if not baseline:
        return None
    step = candidate.get("step_time_s")
    base_step = baseline.get("step_time_s")
    if not isinstance(step,
                      (int, float)) or not isinstance(base_step, (int, float)):
        return None
    if base_step <= 0:
        return None
    delta = (step - base_step) / base_step
    if abs(delta) > step_tolerance:
        return None
    reasons: list[str] = []
    if changes_memory_placement(candidate.get("sharding_overrides"),
                                candidate.get("remat")):
        reasons.append("changes host offload / remat placement")
    for key, label in (("peak_hbm_bytes", "HBM"), ("peak_host_bytes", "host")):
        cand_bytes, base_bytes = candidate.get(key), baseline.get(key)
        if isinstance(cand_bytes, int) and isinstance(base_bytes, int):
            if abs(cand_bytes - base_bytes) >= hbm_delta_bytes:
                reasons.append(f"peak {label} {format_bytes(base_bytes)} ->"
                               f" {format_bytes(cand_bytes)}")
    if not reasons:
        return None
    return (
        f"{SIM_INCONCLUSIVE} (simulated step {delta:+.1%} vs baseline, within"
        f" +/-{step_tolerance:.0%}; {'; '.join(reasons)})")


def check_trial_paths(trial: Mapping[str, Any],
                      plan_paths: Collection[str]) -> None:
    """Raises `UnknownConfigPathError` when the trial writes unknown paths."""
    unknown = find_unknown_paths(trial, plan_paths)
    if unknown:
        raise UnknownConfigPathError(unknown)


# ---------------------------------------------------------------------------
# Formatting.
# ---------------------------------------------------------------------------


def format_bytes(num_bytes: int | None) -> str:
    """Formats a byte count as GiB, or `n/a` when unknown."""
    if num_bytes is None:
        return "n/a"
    return f"{num_bytes / 2**30:.2f} GiB"


def format_collectives(stats: HloStats | None) -> str:
    """Formats collective counts (`AG=2 AR=0 ...`), or `n/a` without stats."""
    if stats is None:
        return "n/a"
    short = {
        "all-gather": "AG",
        "all-reduce": "AR",
        "all-to-all": "A2A",
        "ragged-all-to-all": "RA2A",
        "reduce-scatter": "RS",
        "collective-permute": "CP",
    }
    return (" ".join(f"{short[op]}={stats.collectives.get(op, 0)}"
                     for op in COLLECTIVE_OPCODES) +
            f" (instr={stats.instruction_count})")
