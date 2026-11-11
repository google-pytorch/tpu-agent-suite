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
"""Post-failure log classifier and prior-outcome lookup (`lessons.py`).

Distinguishes XLA per-core VMEM compilation spills (`vmem_oom`, e.g. `68 MiB >
64 MiB` `tpu7x` VMEM on `absorb_q_head` at `DCP >= 16`) from HBM capacity OOMs
(`hbm_oom`), host RAM OOMs (`host_oom`), P/D transfer gate failures, and
cluster node preemptions, and matches the outcome lines close-gaps records in
its ledger and gap files (`ledger.read`) against the settings of a new arm for
`preflight` and `code_gap`.
"""

import dataclasses
import re
from collections.abc import Mapping, Sequence
from typing import Any

from hillclimb import code_gap, ledger, run_history

INFRA = "infra"
CAPACITY = "capacity"
CONFIG = "config"
CODE = "code"
CORRECTNESS = "correctness"
UNKNOWN = "unknown"

_COLLATERAL_EXITS = frozenset({"0", "143"})
_EVIDENCE_CHARS = 300
_EXCEPTION_LINE_RE = re.compile(r"\b[A-Z]\w*(?:Error|Exception): \S")
_VMEM_OP_RE = re.compile(
    r'op_name="([^"]+)"|HLO :: (\S+)|XLA label:\s*([A-Za-z0-9_.]+)\s*=|'
    r"\bin\s+([a-z_][a-z0-9_]*_head|[a-z_]*pallas[a-z0-9_]*)\b")
_VMEM_BYTES_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*([MG]i?B?|M)\b.*?(?:limit|capacity|of|/)\s*"
    r"(\d+(?:\.\d+)?)\s*([MG]i?B?|M)\b",
    re.IGNORECASE,
)


@dataclasses.dataclass(frozen=True)
class FailurePattern:
    """Recognizable failure signature in serving container logs or events."""

    name: str
    category: str
    regexes: tuple[str, ...]
    lesson: str
    retry: str
    preventable: bool


# Ordered most-specific first; `vmem_oom` MUST precede `hbm_oom` because XLA
# VMEM spills also emit `RESOURCE_EXHAUSTED`.
PATTERNS: tuple[FailurePattern, ...] = (
    FailurePattern(
        name="pd_gate_failed",
        category=CORRECTNESS,
        regexes=(r"PD_GATE_FAILED", ),
        lesson=(
            "The P/D correctness gate detected output divergence before the "
            "benchmark: KV transfer, cache layout, or kernel numerics bug."),
        retry=(
            "Do not rerun as is: reproduce the KV transfer or kernel numerics "
            "offline in a unit test and fix it first."),
        preventable=False,
    ),
    FailurePattern(
        name="startup_topology_rejected",
        category=CODE,
        regexes=(
            r"ValueError: .*(ASYM_DCP|Stage-3|DCP interleave|must be divisible "
            r"by dcp_size|admission)", ),
        lesson=(
            "A serving leg rejected its parallelism, bucket divisibility, or "
            "KV transfer geometry at startup."),
        retry=(
            "Fix the bucket/DCP/interleave geometry flagged by preflight and "
            "unit-test the admission check before relaunching."),
        preventable=True,
    ),
    FailurePattern(
        name="vmem_oom",
        category=CONFIG,
        regexes=(
            r"Ran out of memory in memory space vmem",
            r"Exceeded vmem capacity",
            r"(?i)out of memory.{0,50}\bvmem\b",
            r"(?i)\bvmem\b.{0,50}(?:exceed|out of memory|64\s*MiB|63\.94M)",
        ),
        lesson=(
            "XLA compilation exceeded per-core VMEM (64 MiB on tpu7x), NOT "
            "HBM: a Pallas kernel window or scratch buffer is too wide for "
            "this leg's head/token geometry (for example absorb_q_head "
            "allocating 68 MiB > 64 MiB at DCP >= 16)."),
        retry=("Do NOT lower *_KV_BYTES (KV cache lives in HBM, not VMEM): "
               "disable the offending kernel flag on this leg, tile its block "
               "dimension, or lower DECODE_DCP_SIZE <= 8."),
        preventable=True,
    ),
    FailurePattern(
        name="hbm_oom",
        category=CONFIG,
        regexes=(
            r"Ran out of memory in memory space hbm",
            r"(?i)out of memory.{0,40}\bhbm\b",
            r"RESOURCE_EXHAUSTED",
        ),
        lesson=(
            "Device HBM ran out during weight loading, KV cache allocation, "
            "or XLA buffer assignment."),
        retry=(
            "Lower PREFILL_KV_BYTES / DECODE_KV_BYTES / gpu_memory_utilization "
            "or disable flags whose documentation warns about extra HBM."),
        preventable=True,
    ),
    FailurePattern(
        name="host_oom",
        category=CONFIG,
        regexes=(
            r"OOMKilled",
            r"Out of memory: Killed process",
            r"Killed process \d+",
        ),
        lesson=(
            "Host DRAM ran out (oversized CPU KV offload tier, staging leases, "
            "or small prefill compile buckets)."),
        retry=("Lower PREFILL_KV_OFFLOAD_GIB, keep *_STAGING_LEASES <= 2, or "
               "remove prefill compile buckets < 512."),
        preventable=True,
    ),
    FailurePattern(
        name="tpu_session",
        category=INFRA,
        regexes=(
            r"START_SESSION failed",
            r"MaybeInitializeSliceBuilder failed",
            r"FATAL_TPU_SESSION",
            r"Cannot write the request to vBAR",
        ),
        lesson="The TPU runtime session failed to initialize on a slice host.",
        retry=
        "Transient hardware/runtime failure: retry unchanged on a healthy slice.",
        preventable=False,
    ),
    FailurePattern(
        name="node_failure",
        category=INFRA,
        regexes=(
            r"EvictedDueToNodeFailures",
            r"TerminationByKubelet",
            r"NodeNotReady",
            r"node preemption detected",
            r"\bPreempted\b",
            r"assigned node failures",
            r"no replacement for unhealthy node",
            r"TaintManagerEviction",
        ),
        lesson="A cluster node in the slice was preempted or evicted mid-run.",
        retry="Infrastructure eviction: retry unchanged.",
        preventable=False,
    ),
    FailurePattern(
        name="pending_timeout",
        category=CAPACITY,
        regexes=(
            r"(?i)pending[ _-]?timeout",
            r"(?i)timed out .{0,40}\b(pending|suspended)\b",
        ),
        lesson=
        "Cluster queue did not allocate the requested slice before timeout.",
        retry=
        "Retry with a longer queue timeout or validate on a 1-host microbenchmark first.",
        preventable=False,
    ),
    FailurePattern(
        name="test_failure",
        category=CODE,
        regexes=(r"=+ .*\b\d+ failed\b", ),
        lesson="Pytest microbenchmark or unit test assertions failed.",
        retry=
        "Fix the failing kernel or test case locally before multi-slice runs.",
        preventable=True,
    ),
    FailurePattern(
        name="python_error",
        category=CODE,
        regexes=(r"Traceback \(most recent call last\)", ),
        lesson=
        "A server or benchmark process raised an unhandled Python exception.",
        retry=
        "Fix the exception in the traceback and cover it with a unit test.",
        preventable=True,
    ),
)

_UNMATCHED = FailurePattern(
    name="unknown",
    category=UNKNOWN,
    regexes=(),
    lesson="No registered failure pattern matched the provided log.",
    retry=
    "Inspect the failing container log tail and add a deterministic pattern.",
    preventable=False,
)


@dataclasses.dataclass(frozen=True)
class Diagnosis:
    """Structured failure classification for a failed serving run."""

    run_id: str
    pattern: str
    category: str
    preventable: bool
    evidence: str
    lesson: str
    retry: str
    offending_op: str = ""
    container: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _extract_vmem_op(text: str) -> str:
    for m in _VMEM_OP_RE.finditer(text):
        for grp in m.groups():
            if grp:
                return grp.strip()
    return ""


def _extract_evidence(text: str, regex: str) -> str:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if re.search(regex, line):
            # If it's a Python traceback header, prefer the final exception line.
            if "Traceback (most recent call last)" in line:
                for tail in reversed(lines[idx:]):
                    if _EXCEPTION_LINE_RE.search(tail):
                        return tail.strip()[:_EVIDENCE_CHARS]
            return line.strip()[:_EVIDENCE_CHARS]
    m = re.search(regex, text)
    return m.group(0).strip()[:_EVIDENCE_CHARS] if m else ""


def diagnose_log(
        log_text: str,
        *,
        run_id: str = "failed-run",
        container: str = "",
        events: Sequence[str] = (),
) -> Diagnosis:
    """Classifies a raw log string (plus optional cluster events)."""
    combined = "\n".join([log_text, *events])
    for pat in PATTERNS:
        for rgx in pat.regexes:
            if re.search(rgx, combined):
                ev = _extract_evidence(combined, rgx)
                op = _extract_vmem_op(
                    combined) if pat.name == "vmem_oom" else ""
                lesson = pat.lesson
                if op and op not in lesson:
                    lesson = f"{lesson} Offending op: `{op}`."
                m_bytes = _VMEM_BYTES_RE.search(
                    combined) if pat.name == "vmem_oom" else None
                if m_bytes and op:
                    ev = f"{ev} (op={op})"
                return Diagnosis(
                    run_id=run_id,
                    pattern=pat.name,
                    category=pat.category,
                    preventable=pat.preventable,
                    evidence=ev,
                    lesson=lesson,
                    retry=pat.retry,
                    offending_op=op,
                    container=container,
                )
    return Diagnosis(
        run_id=run_id,
        pattern=_UNMATCHED.name,
        category=_UNMATCHED.category,
        preventable=_UNMATCHED.preventable,
        evidence=log_text.strip().splitlines()[-1][:_EVIDENCE_CHARS]
        if log_text.strip() else "",
        lesson=_UNMATCHED.lesson,
        retry=_UNMATCHED.retry,
        container=container,
    )


def diagnose(
    run_desc: Mapping[str, Any],
    logs: Mapping[str, str],
) -> Diagnosis:
    """Diagnoses a failed run metadata dictionary and `container -> log_tail` map."""
    run_id = str(run_desc.get("run_id") or run_desc.get("id") or "failed-run")
    raw_events = run_desc.get("events") or ()
    events = [
        str(e.get("message", e)) if isinstance(e, Mapping) else str(e)
        for e in raw_events
    ]
    for container, text in logs.items():
        diag = diagnose_log(text,
                            run_id=run_id,
                            container=container,
                            events=events)
        if diag.pattern != "unknown":
            return diag
    first_container = next(iter(logs), "")
    first_text = logs.get(first_container, "")
    return diagnose_log(first_text,
                        run_id=run_id,
                        container=first_container,
                        events=events)


def render_diagnosis(diag: Diagnosis) -> str:
    """Formats a `Diagnosis` as readable Markdown."""
    prev = "yes (preflight/unit-test)" if diag.preventable else "no (runtime/cluster)"
    lines = [
        f"### Failure Diagnosis: `{diag.run_id}` -> **{diag.pattern}** (`{diag.category}`)",
        "",
        f"- **Preventable before launch**: {prev}",
    ]
    if diag.container:
        lines.append(f"- **Container**: `{diag.container}`")
    if diag.offending_op:
        lines.append(f"- **Offending HLO/Pallas op**: `{diag.offending_op}`")
    if diag.evidence:
        lines.append(f"- **Evidence**: `{diag.evidence}`")
    lines.append(f"- **Lesson**: {diag.lesson}")
    lines.append(f"- **Next action**: {diag.retry}")
    return "\n".join(lines) + "\n"


def settings_diff(
    arm_knobs: Mapping[str, str],
    base_knobs: Mapping[str, str],
    defaults: Mapping[str, str] | None = None,
    conventions: run_history.KnobConventions = run_history.DEFAULT_CONVENTIONS,
) -> list[str]:
    """Returns canonical `KEY=VALUE` / `ENV=VALUE` settings changed vs baseline."""
    defs = defaults or {}
    arm_p = run_history.perf_knobs(arm_knobs, defs, conventions)
    base_p = run_history.perf_knobs(base_knobs, defs, conventions)
    extra_keys = set(conventions.extra_env)

    diffs: list[str] = []
    for key in sorted(set(arm_p) | set(base_p)):
        if key in extra_keys:
            a_env = code_gap.parse_extra_env(arm_p.get(key, ""))
            b_env = code_gap.parse_extra_env(base_p.get(key, ""))
            for flag in sorted(set(a_env) | set(b_env)):
                if a_env.get(flag) != b_env.get(flag):
                    diffs.append(f"{flag}={a_env.get(flag, '')}")
        else:
            if arm_p.get(key) != base_p.get(key):
                diffs.append(f"{key}={arm_p.get(key, '')}")
    return diffs


@dataclasses.dataclass(frozen=True)
class PriorOutcome:
    """Past recorded outcome matching one or more settings of a new arm."""

    run_id: str
    baseline: str
    decision: str
    delta_rel: float | None
    settings: tuple[str, ...]
    lesson: str = ""
    # Settings the recorded outcome changed in total. Above 1, the verdict
    # belongs to that bundle, not to any single overlapping setting.
    bundle_size: int = 1


def prior_outcomes(
    entries: Sequence[ledger.Entry],
    added_settings: Sequence[str],
) -> list[PriorOutcome]:
    """Finds recorded close-gaps outcomes overlapping `added_settings`."""
    want = set(added_settings)
    if not want:
        return []
    found: list[PriorOutcome] = []
    for entry in reversed(entries):
        if entry.kind != "decision":
            continue
        p = entry.payload
        recorded_settings = tuple(str(s) for s in (p.get("settings") or ()))
        overlap = tuple(s for s in recorded_settings if s in want)
        if not overlap:
            continue
        raw_delta = p.get("delta_rel")
        delta = (float(raw_delta) if isinstance(raw_delta, (int, float))
                 and not isinstance(raw_delta, bool) else None)
        found.append(
            PriorOutcome(
                run_id=str(p.get("run_id") or p.get("job") or ""),
                baseline=str(p.get("baseline") or ""),
                decision=str(p.get("decision") or ""),
                delta_rel=delta,
                settings=overlap,
                lesson=str(p.get("lesson") or p.get("reason") or ""),
                bundle_size=max(1, len(recorded_settings)),
            ))
    return found


def flag_verdicts(
    entries: Sequence[ledger.Entry], ) -> dict[str, code_gap.PriorVerdict]:
    """Maps `FLAG_NAME -> latest PriorVerdict` from single-setting outcomes."""
    out: dict[str, code_gap.PriorVerdict] = {}
    for entry in entries:
        if entry.kind != "decision":
            continue
        p = entry.payload
        settings = [str(s) for s in (p.get("settings") or ())]
        if len(settings) != 1 or "=" not in settings[0]:
            continue
        flag, val = settings[0].split("=", 1)
        if not val:
            continue
        raw_delta = p.get("delta_rel")
        delta = (float(raw_delta) if isinstance(raw_delta, (int, float))
                 and not isinstance(raw_delta, bool) else None)
        out[flag] = code_gap.PriorVerdict(
            decision=str(p.get("decision") or ""),
            delta_rel=delta,
            run_id=str(p.get("run_id") or p.get("job") or ""),
            lesson=str(p.get("lesson") or p.get("reason") or ""),
        )
    return out
