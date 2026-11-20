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
"""Benchmark telemetry extractor and trajectory auditor.

Extracts performance telemetry (parity/correctness, speedup multiplier, latency)
across both historical ATIF trajectory logs and modern standardized results files.
"""

import json
import os
import re
from pathlib import Path
from typing import Any


def _logs_dir() -> Path:
    """The run's logs as the harness lays them out: $LOGS_DIR, else /logs."""
    return Path(os.environ.get("LOGS_DIR") or "/logs")


def _trajectory_path() -> Path:
    """The agent's trajectory, where the harness writes it."""
    return _logs_dir() / "agent" / "trajectory.json"


def _load_trajectory(path: str | Path) -> dict[str, Any] | None:
    """Load an ATIF trajectory JSON file. Returns None on error."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def find_results_artifact(workspace: Path) -> Path | None:
    """Finds results.json or alternative result artifacts in the workspace."""
    search_names = (
        "results.json",
        "result.json",
        "report.json",
        "metrics.json",
        "benchmark_results.json",
        "benchmark_report.json",
    )
    for name in search_names:
        cand = workspace / name
        if cand.is_file() and cand.stat().st_size > 0:
            return cand

    try:
        for p in workspace.iterdir():
            if p.is_file() and p.suffix.lower() == ".json":
                if any(
                        n.replace(".json", "") in p.stem.lower()
                        for n in search_names):
                    return p
    except OSError:
        pass
    return None


def extract_report(
    workspace: Path,
    target_speedup: float | None = None,
    baseline_latency_ms: float | None = None,
    custom_parity_patterns: list[str] | None = None,
    custom_speedup_patterns: list[str] | None = None,
    disqualifier_patterns: list[str] | None = None,
) -> dict[str, Any]:
    """Audits workspace artifacts and agent trajectory across historical & new formats.

    Args:
        workspace: Path to candidate workspace.
        target_speedup: Target speedup multiplier threshold (e.g. 1.0 or 1.5).
        baseline_latency_ms: Reference baseline p50 latency in ms.
        custom_parity_patterns: Optional task-specific regex patterns indicating correctness.
        custom_speedup_patterns: Optional task-specific regex patterns for speedup extraction.
        disqualifier_patterns: Optional regex patterns that invalidate correctness.

    Returns:
        dict containing 'correctness' and 'speedup' sub-dictionaries.
    """
    eff_target_speedup = float(
        target_speedup) if target_speedup is not None else 1.0
    eff_base_ms = float(
        baseline_latency_ms) if baseline_latency_ms is not None else None

    # 1. Check for results.json (new standard reporting contract)
    results_file = find_results_artifact(workspace)
    results_json: dict[str, Any] | None = None
    if results_file:
        try:
            raw_data = json.loads(results_file.read_text(encoding="utf-8"))
            if isinstance(raw_data, dict):
                results_json = raw_data
        except Exception:
            pass

    # 2. Build corpus from workspace files
    corpus: dict[str, str] = {}
    try:
        for f in workspace.rglob("*"):
            if f.is_file(
            ) and ".git" not in f.parts and "__pycache__" not in f.parts:
                try:
                    corpus[f.name] = f.read_text(encoding="utf-8",
                                                 errors="ignore")[:250_000]
                except OSError:
                    pass
    except Exception:
        pass

    # 3. Read the trajectory from the harness's agent logs
    agent_text = ""
    final_assistant_message = ""
    tp = _trajectory_path()
    if tp.is_file():
        try:
            traj_data = _load_trajectory(tp)
            if isinstance(traj_data, dict):
                for step in reversed(traj_data.get("steps", [])):
                    msg = step.get("content") or step.get("message") or ""
                    if isinstance(msg, str) and len(msg.strip()) > 50:
                        final_assistant_message = msg
                        break
                for step in traj_data.get("steps", []):
                    for tc in step.get("tool_calls", []):
                        cmd = (tc.get("arguments", {}).get("command")
                               or tc.get("arguments", {}).get("CommandLine")
                               or "")
                        agent_text += f"\n[CMD] {cmd}"
                    for obs in step.get("observations", []):
                        content = obs.get("content") or obs.get("output") or ""
                        agent_text += f"\n[OUTPUT] {content}"
                    msg_content = step.get("content") or step.get(
                        "message") or ""
                    if isinstance(msg_content, str):
                        agent_text += f"\n[MSG] {msg_content}"
                        if not results_json:
                            json_match = re.search(
                                r"```json\s*(\{.*?\})\s*```", msg_content,
                                re.DOTALL)
                            if json_match:
                                try:
                                    parsed = json.loads(json_match.group(1))
                                    if isinstance(parsed, dict) and (
                                            "correctness" in parsed
                                            or "speedup_factor" in parsed):
                                        results_json = parsed
                                except Exception:
                                    pass
        except Exception:
            pass

    # Fallback: scan any text/log files in agent log directories
    agent_dir_resolved = _logs_dir() / "agent"
    if agent_dir_resolved.is_dir():
        for log_f in sorted(agent_dir_resolved.rglob("*"))[:200]:
            if log_f.is_file() and log_f.suffix in (".log", ".txt", ".json",
                                                    ".jsonl"):
                try:
                    agent_text += "\n" + log_f.read_text(
                        "utf-8", errors="ignore")[:100_000]
                except OSError:
                    pass

    blob = "\n".join(corpus.values()) + "\n" + agent_text

    # =========================================================================
    # 4. NUMERICAL ACCURACY / PARITY EXTRACTION
    # =========================================================================
    correctness_passed = False
    correctness_reason = "No verification evidence found"
    max_diff: float | None = None

    if results_json:
        corr_val = results_json.get("correctness")
        if isinstance(corr_val, dict):
            correctness_passed = bool(corr_val.get("passed", False))
            max_diff = corr_val.get("max_diff") or corr_val.get(
                "max_absolute_difference")
        elif isinstance(corr_val, bool):
            correctness_passed = corr_val
            max_diff = results_json.get("max_diff")
        if correctness_passed:
            correctness_reason = f"results.json reported correctness={correctness_passed}"

    # Tier 1: Explicit Parity Gate string (e.g. "PARITY GATE 0.069888 -> PASS" or "PARITY: PASS" or "parity gate = ... — solution passes")
    search_target = f"{final_assistant_message}\n{blob}" if final_assistant_message else blob
    if not correctness_passed:
        gate_match = re.search(
            r"PARITY(?:\s*GATE)?(?:\s+([\d\.\-e]+))?[^\n]*->\s*(PASS|FAIL)\b|\bPARITY:\s*(PASS|FAIL)\b|parity\s*gate\s*=\s*[^\n]*—\s*solution\s*(passes|fails)\b",
            search_target,
            re.I,
        )
        if gate_match:
            diff_str = gate_match.group(1)
            verdict = gate_match.group(2) or gate_match.group(
                3) or gate_match.group(4)
            if diff_str and max_diff is None:
                try:
                    max_diff = float(diff_str)
                except ValueError:
                    pass
            if verdict and verdict.lower() in ("pass", "passes"):
                correctness_passed = True
                correctness_reason = f"Explicit marker: {gate_match.group(0)}"

    # Tier 2: Explicit error measurements
    diff_match = re.search(r"CORRECTNESS\[solution\][^\n:]*:\s*([\d\.\-e]+)",
                           blob, re.I)
    if diff_match:
        try:
            max_diff = float(diff_match.group(1))
        except ValueError:
            pass

    if max_diff is None and final_assistant_message:
        m_tbl_diff = re.search(
            r"\|\s*(?:\*\*)?solution(?:\*\*)?\s*\|[^\n]*\|\s*(?:\*\*)?([\d\.\-e]+)\s*(?:\*\*)?\s*\|",
            final_assistant_message,
            re.I,
        )
        if m_tbl_diff:
            try:
                max_diff = float(m_tbl_diff.group(1))
            except ValueError:
                pass

    if max_diff is None:
        diff_match2 = re.search(
            r"max(?:imum)?\s*(?:abs(?:olute)?)?\s*diff(?:erence)?[^\n\d]*([\d\.\-e]+)",
            blob, re.I)
        if diff_match2:
            try:
                max_diff = float(diff_match2.group(1))
            except ValueError:
                pass

    # Tier 3: Custom task-specific parity patterns
    if not correctness_passed and custom_parity_patterns:
        for pat in custom_parity_patterns:
            m = re.search(pat, blob, re.I)
            if m:
                correctness_passed = True
                correctness_reason = f"Custom parity match: {m.group(0)}"
                break

    # Tier 4: Prose confirmation in output
    if not correctness_passed:
        if re.search(
                r"(?:solution|parity|kernel)\s+passes\b|passed\s+all\s+(?:parity|tests)|more\s+accurate\s+than\s+the\s+reference",
                blob,
                re.I,
        ):
            correctness_passed = True
            correctness_reason = "Prose parity pass confirmation detected in output"

    # Tier 5: Test execution with zero assertion errors
    if not correctness_passed:
        if re.search(r"(?:python3?\s+.*(?:verify|check|test|parity)|pytest)",
                     agent_text, re.I):
            if not re.search(
                    r"(?:AssertionError|FAILED\s*\(failures=\d+\)|Traceback\s*\(most\s*recent)",
                    agent_text):
                correctness_passed = True
                correctness_reason = "Verification test script executed with exit 0 and zero assertion errors"

    # Disqualifiers: Hard assertion failure, NaN, or custom disqualifiers
    disqualifier_list = [
        r"PARITY(?:\s*GATE)?[^\n]*->\s*FAIL\b",
        r"AssertionError:\s*(?:get_inputs\s*mismatch|parity\s*failed|allclose\s*failed)",
        r"AssertionError",
    ]
    if disqualifier_patterns:
        disqualifier_list.extend(disqualifier_patterns)

    for disq in disqualifier_list:
        if re.search(disq, agent_text, re.I):
            correctness_passed = False
            correctness_reason = f"Disqualified: Hard failure matched pattern '{disq}'"
            break

    # =========================================================================
    # 5. SPEEDUP / MULTIPLIER EXTRACTION
    # =========================================================================
    speedup_factor: float | None = None
    cand_ms: float | None = None
    base_ms = eff_base_ms
    speedup_source = "not_found"

    if results_json:
        speedup_factor = results_json.get(
            "speedup_factor") or results_json.get("speedup")
        cand_ms = results_json.get("candidate_latency_ms") or results_json.get(
            "best_latency_ms")
        if cand_ms is None and "candidate_time_us" in results_json:
            cand_ms = float(results_json["candidate_time_us"]) / 1000.0

        base_ms = results_json.get("baseline_latency_ms")
        if base_ms is None and "baseline_time_us" in results_json:
            base_ms = float(results_json["baseline_time_us"]) / 1000.0
        if base_ms is None:
            base_ms = eff_base_ms

        if speedup_factor is not None:
            speedup_source = "results_json"

    # Priority Tier: Final assistant message (the definitive final deliverable)
    if final_assistant_message:
        if speedup_factor is None:
            m_tag_f = re.search(
                r"SPEEDUP:\s*([\d\.]+)[x×](?:\s+RESULT_TIME_MS:\s*([\d\.]+))?",
                final_assistant_message, re.I)
            if m_tag_f:
                speedup_factor = float(m_tag_f.group(1))
                speedup_source = "final_msg:explicit_tag"
                if m_tag_f.group(2):
                    cand_ms = float(m_tag_f.group(2))

        if speedup_factor is None:
            m_prose_f = re.search(r"([\d\.]+)\s*[x×]\s*(?:faster|speedup)",
                                  final_assistant_message, re.I)
            if m_prose_f:
                speedup_factor = float(m_prose_f.group(1))
                speedup_source = "final_msg:prose_multiplier"

        if cand_ms is None:
            m_tbl_cand_f = re.search(
                r"\|\s*(?:\*\*)?(?:solution|this kernel|optimized)(?:\*\*)?\s*\|\s*(?:\*\*)?([\d\.]+)\s*ms",
                final_assistant_message, re.I)
            if m_tbl_cand_f:
                cand_ms = float(m_tbl_cand_f.group(1))

        m_tbl_base_f = re.search(
            r"\|\s*(?:\*\*)?(?:reference[^\n|]*|base\.py)(?:\*\*)?\s*\|\s*(?:\*\*)?([\d\.]+)\s*ms",
            final_assistant_message, re.I)
        if m_tbl_base_f:
            base_ms = float(m_tbl_base_f.group(1))

    # Tier 1: Custom speedup patterns
    if speedup_factor is None and custom_speedup_patterns:
        for pat in custom_speedup_patterns:
            m = re.search(pat, blob, re.I)
            if m:
                try:
                    speedup_factor = float(m.group(1))
                    speedup_source = f"custom_pattern: {pat}"
                    break
                except ValueError:
                    pass

    # Tier 2: Explicit SPEEDUP tag (e.g. "SPEEDUP: 5.43x   RESULT_TIME_MS: 2.997")
    if speedup_factor is None:
        m_tag = re.search(
            r"SPEEDUP:\s*([\d\.]+)[x×](?:\s+RESULT_TIME_MS:\s*([\d\.]+))?",
            blob, re.I)
        if m_tag:
            speedup_factor = float(m_tag.group(1))
            speedup_source = "explicit_speedup_tag"
            if m_tag.group(2):
                cand_ms = float(m_tag.group(2))

    # Tier 3: Prose multiplier statement (e.g. "5.43× faster" or "5.43x speedup")
    if speedup_factor is None:
        m_prose = re.search(r"([\d\.]+)\s*[x×]\s*(?:faster|speedup)", blob,
                            re.I)
        if m_prose:
            speedup_factor = float(m_prose.group(1))
            speedup_source = "prose_multiplier"

    # Tier 4: Terminal p50 lines (e.g. "[solution] p50=2.997ms")
    if cand_ms is None:
        m_cand = re.search(
            r"\[(?:solution|optimized|kernel)\]\s*p50=([\d\.]+)ms", blob, re.I)
        m_base = re.search(r"\[(?:reference|base)[^\]]*\]\s*p50=([\d\.]+)ms",
                           blob, re.I)
        if m_cand:
            cand_ms = float(m_cand.group(1))
        if m_base:
            base_ms = float(m_base.group(1))

    # Tier 5: Markdown comparison table rows
    if cand_ms is None:
        m_tbl_cand = re.search(
            r"\|\s*(?:\*\*)?(?:solution|this kernel|optimized)(?:\*\*)?\s*\|\s*(?:\*\*)?([\d\.]+)\s*ms",
            blob, re.I)
        m_tbl_base = re.search(
            r"\|\s*(?:\*\*)?(?:reference[^\n|]*|base\.py)(?:\*\*)?\s*\|\s*(?:\*\*)?([\d\.]+)\s*ms",
            blob, re.I)
        if m_tbl_cand:
            cand_ms = float(m_tbl_cand.group(1))
        if m_tbl_base:
            base_ms = float(m_tbl_base.group(1))

    # Tier 6: Docstring in solution.py (e.g. "-> 5.43x faster")
    if speedup_factor is None:
        m_doc = re.search(r"->\s*([\d\.]+)[x×]\s*faster", blob, re.I)
        if m_doc:
            speedup_factor = float(m_doc.group(1))
            speedup_source = "docstring_multiplier"

    # Multiplier calculation fallback from latencies
    if speedup_factor is None and cand_ms and base_ms:
        if cand_ms > 0:
            speedup_factor = round(base_ms / cand_ms, 2)
            speedup_source = f"calculated_from_latencies ({base_ms}ms / {cand_ms}ms)"

    passed_speedup = False
    if correctness_passed and speedup_factor is not None:
        passed_speedup = bool(speedup_factor >= eff_target_speedup)

    return {
        "correctness": {
            "passed": correctness_passed,
            "max_diff": max_diff,
            "reason": correctness_reason,
        },
        "speedup": {
            "passed": passed_speedup,
            "speedup_factor": speedup_factor,
            "candidate_latency_ms": cand_ms,
            "baseline_latency_ms": base_ms,
            "target_speedup": eff_target_speedup,
            "source": speedup_source,
        },
    }
