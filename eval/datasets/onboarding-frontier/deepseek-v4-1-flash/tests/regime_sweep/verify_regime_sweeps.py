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
"""Deterministic benchmark regime-sweep verifier for onboarding-only tasks.

Adapted from PR #694 (`calculate_speedup.py`). Parses benchmark sweep JSONs,
CSVs, and Markdown tables across the 3 required ISL/OSL regimes (`1k/1k`,
`1k/8k`, `8k/1k`) from `/logs/artifacts` (or `<trial_dir>/artifacts`) and
falls back to scanning `agent/trajectory.json` for runs that timed out before
staging final artifacts.
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple

try:
    from rewardkit import criterion
except ImportError:

    def criterion(**kwargs):  # type: ignore[misc]

        def _decorator(fn):
            return fn

        return _decorator


Path = pathlib.Path

REQUIRED_REGIMES = ("1k/1k", "1k/8k", "8k/1k")
TWO_REGIME_REQUIRED = ("1k/1k", "8k/1k")


def detect_required_regimes(workspace: Optional[Path]) -> Tuple[str, ...]:
    """Detects whether the task prompt requested 2 regimes (1k/1k, 8k/1k) or 3 regimes (+ 1k/8k)."""
    traj_candidates: List[Path] = []
    if os.environ.get("ATIF_TRAJECTORY"):
        traj_candidates.append(Path(os.environ["ATIF_TRAJECTORY"]))
    if os.environ.get("TRAJECTORY_PATH"):
        traj_candidates.append(Path(os.environ["TRAJECTORY_PATH"]))
    if workspace:
        traj_candidates.extend([
            workspace / "agent" / "trajectory.json",
            workspace / "trajectory.json",
        ])
    traj_candidates.append(Path("/logs/agent/trajectory.json"))

    for tc in traj_candidates:
        if tc and tc.is_file():
            try:
                traj = json.loads(
                    tc.read_text(encoding="utf-8", errors="ignore"))
                for step in (traj.get("steps") or [])[:5]:
                    if not isinstance(step, dict):
                        continue
                    msg = str(step.get("message") or "").lower()
                    if "input/output length" in msg or "isl" in msg or "1024" in msg:
                        if any(k in msg
                               for k in ("1k/8k", "1024/8192", "1024 / 8192",
                                         "isl=1024, osl=8192",
                                         "isl 1024, osl 8192")):
                            return REQUIRED_REGIMES
                        if any(k in msg
                               for k in ("8k/1k", "8192/1024", "8192 / 1024",
                                         "isl=8192, osl=1024",
                                         "isl 8192, osl 1024")):
                            return TWO_REGIME_REQUIRED
            except Exception:
                pass
    return REQUIRED_REGIMES


def normalize_regime_name(name: str) -> str:
    """Normalizes variations like '1k1k', '1k_1k', '1024_1024', 'balanced' into '1k/1k'."""
    clean = re.sub(r"\s*([/:-])\s*", "/",
                   str(name).lower().replace("-", "/").replace("_",
                                                               "/")).strip()
    if any(k in clean
           for k in ("1k/1k", "1k1k", "1024/1024", "isl1024/osl1024",
                     "isl 1024, osl 1024", "isl=1024, osl=1024", "balanced")):
        return "1k/1k"
    if any(k in clean
           for k in ("8k/1k", "8k1k", "8192/1024", "isl8192/osl1024",
                     "isl 8192, osl 1024", "isl=8192, osl=1024", "prefill")):
        return "8k/1k"
    if any(k in clean
           for k in ("1k/8k", "1k8k", "1024/8192", "isl1024/osl8192",
                     "isl 1024, osl 8192", "isl=1024, osl=8192", "decode")):
        return "1k/8k"
    return clean


class BenchmarkRunMetrics:
    """Telemetry metrics extracted from a single benchmark concurrency point."""

    def __init__(
        self,
        regime: str,
        concurrency: int,
        output_tps: float,
        duration_s: float = 0.0,
        completed: int = 0,
        failed: int = 0,
        ttft_p50: Optional[float] = None,
        itl_p50: Optional[float] = None,
        source: str = "",
        notes: str = "",
    ):
        self.regime = normalize_regime_name(regime)
        self.concurrency = int(concurrency)
        self.output_tps = float(output_tps)
        self.duration_s = float(duration_s)
        self.completed = int(completed)
        self.failed = int(failed)
        self.ttft_p50 = float(ttft_p50) if ttft_p50 is not None else None
        self.itl_p50 = float(itl_p50) if itl_p50 is not None else None
        self.source = str(source)
        self.notes = str(notes)

    @classmethod
    def from_dict(
        cls,
        raw: Dict[str, Any],
        default_regime: str = "1k/1k",
        default_concurrency: Optional[int] = None,
        source: str = "",
    ) -> Optional[BenchmarkRunMetrics]:
        """Parses a benchmark result dictionary from vllm bench serve, vmbench, or summary JSON."""
        s = raw.get("summary") if isinstance(raw.get("summary"), dict) else raw

        regime = (s.get("regime") or raw.get("regime") or s.get("isl_osl")
                  or raw.get("isl_osl") or default_regime)
        isl = (s.get("isl") or raw.get("isl") or s.get("input_len")
               or raw.get("input_len") or s.get("random_input_len")
               or raw.get("random_input_len"))
        osl = (s.get("osl") or raw.get("osl") or s.get("output_len")
               or raw.get("output_len") or s.get("random_output_len")
               or raw.get("random_output_len"))
        try:
            if isl and osl:
                isl_i, osl_i = int(isl), int(osl)
                if isl_i == 1024 and osl_i == 1024:
                    regime = "1k/1k"
                elif isl_i == 8192 and osl_i == 1024:
                    regime = "8k/1k"
                elif isl_i == 1024 and osl_i == 8192:
                    regime = "1k/8k"
        except (ValueError, TypeError):
            pass

        concurrency = (s.get("concurrency") or raw.get("concurrency")
                       or s.get("max_concurrency")
                       or raw.get("max_concurrency") or default_concurrency
                       or 1)
        if isinstance(concurrency, str):
            c_m = re.search(r"\b(\d+)\b", concurrency)
            concurrency = int(c_m.group(1)) if c_m else 1
        try:
            concurrency = int(concurrency)
        except (ValueError, TypeError):
            concurrency = 1

        output_tps = (
            s.get("output_throughput_tok_s") or s.get("output_tok_per_s")
            or s.get("throughput_tok_per_s")
            or s.get("throughput_tokens_per_s")
            or s.get("output_throughput_tok_per_sec")
            or s.get("throughput_tok_s") or s.get("output_throughput")
            or s.get("output_tokens_per_s") or s.get("throughput")
            or raw.get("output_throughput_tok_s")
            or raw.get("output_tok_per_s") or raw.get("throughput_tok_per_s")
            or raw.get("throughput_tokens_per_s")
            or raw.get("output_throughput_tok_per_sec")
            or raw.get("throughput_tok_s") or raw.get("output_throughput")
            or raw.get("output_tokens_per_s") or raw.get("throughput") or 0.0)
        try:
            output_tps = float(output_tps)
        except (ValueError, TypeError):
            output_tps = 0.0

        if output_tps <= 0.0:
            return None

        ttft_p50 = (s.get("ttft_ms_median") or s.get("median_ttft_ms")
                    or s.get("p50_ttft_ms") or s.get("ttft_ms")
                    or s.get("ttft_p50_ms") or raw.get("median_ttft_ms")
                    or raw.get("ttft_ms"))
        itl_p50 = (s.get("itl_ms_median") or s.get("median_itl_ms")
                   or s.get("p50_itl_ms") or s.get("itl_ms")
                   or s.get("itl_p50_ms") or raw.get("median_itl_ms")
                   or raw.get("itl_ms"))
        try:
            ttft_p50 = float(ttft_p50) if ttft_p50 is not None else None
        except (ValueError, TypeError):
            ttft_p50 = None
        try:
            itl_p50 = float(itl_p50) if itl_p50 is not None else None
        except (ValueError, TypeError):
            itl_p50 = None

        return cls(
            regime=regime,
            concurrency=concurrency,
            output_tps=output_tps,
            ttft_p50=ttft_p50,
            itl_p50=itl_p50,
            source=source,
        )


def infer_regime_and_conc_from_path(path: Path) -> Tuple[str, Optional[int]]:
    """Infers regime ('1k/1k', '1k/8k', '8k/1k') and concurrency from full path string."""
    s = str(path).lower()
    regime = "1k/1k"
    if any(k in s for k in ("isl8192_osl1024", "8192_1024", "8192x1024",
                            "8k1k", "8k_1k", "8k-1k")):
        regime = "8k/1k"
    elif any(k in s for k in ("isl1024_osl8192", "1024_8192", "1024x8192",
                              "1k8k", "1k_8k", "1k-8k")):
        regime = "1k/8k"
    elif any(k in s for k in ("isl1024_osl1024", "1024_1024", "1024x1024",
                              "1k1k", "1k_1k", "1k-1k")):
        regime = "1k/1k"

    conc: Optional[int] = None
    m = re.search(r"[_/-]c(\d+)(?:\b|[_/.-])", s)
    if m:
        conc = int(m.group(1))
    return regime, conc


def parse_markdown_tables(content: str,
                          filename: str) -> List[BenchmarkRunMetrics]:
    """Parses benchmark sweep tables from markdown reports."""
    runs: List[BenchmarkRunMetrics] = []
    current_regime = "1k/1k"
    header_map: Dict[str, int] = {}

    for line in content.splitlines():
        line_clean = line.strip()
        if not line_clean.startswith("|") or not line_clean.endswith("|"):
            header_map = {}
            if line_clean.startswith("#") or (
                    len(line_clean) < 100 and
                (line_clean.startswith("**") or "regime" in line_clean.lower()
                 or "isl" in line_clean.lower())):
                matched_regs = [
                    r for r in REQUIRED_REGIMES
                    if r in normalize_regime_name(line_clean)
                ]
                if len(matched_regs) == 1:
                    current_regime = matched_regs[0]
            continue

        cells = [c.strip() for c in line_clean.split("|")[1:-1]]
        if not cells or all(set(c) <= set("-: ") for c in cells):
            continue

        lower_cells = [c.lower() for c in cells]
        if any("throughput" in c or "tok/s" in c or "tps" in c
               for c in lower_cells) and any(
                   "concurrency" in c or c in ("conc", "c")
                   for c in lower_cells):
            header_map = {}
            for idx, c in enumerate(lower_cells):
                if "concurrency" in c or c in ("conc", "c"):
                    header_map["concurrency"] = idx
                elif ("throughput" in c or "tok/s" in c or "tps"
                      in c) and ("total" not in c and "input" not in c
                                 and "prompt" not in c and "chip" not in c
                                 and "per" not in c):
                    if "throughput" not in header_map or "output" in c:
                        header_map["throughput"] = idx
                elif "ttft" in c:
                    header_map["ttft"] = idx
                elif "itl" in c:
                    header_map["itl"] = idx
                elif any(k in c for k in ("isl/osl", "isl / osl", "regime",
                                          "combination", "workload")):
                    header_map["regime"] = idx
            continue

        if "throughput" in header_map and "concurrency" in header_map:
            clean_cells = [
                re.sub(r"\*\*([^*]+)\*\*", r"\1",
                       c).replace(",", "").replace("*", "").strip()
                for c in cells
            ]
            if len(clean_cells) <= max(header_map.values()):
                continue
            row_regime = current_regime
            if "regime" in header_map and clean_cells[header_map["regime"]]:
                row_regime = normalize_regime_name(
                    clean_cells[header_map["regime"]])
            c_match = re.search(r"\b(\d+)\b",
                                clean_cells[header_map["concurrency"]])
            tps_str = clean_cells[header_map["throughput"]].split(
            )[0] if clean_cells[header_map["throughput"]] else ""
            if c_match and tps_str:
                try:
                    c_val = int(c_match.group(1))
                    tps_val = float(tps_str)
                    if tps_val > 0 and row_regime in REQUIRED_REGIMES:
                        ttft = None
                        itl = None
                        if "ttft" in header_map and clean_cells[
                                header_map["ttft"]]:
                            try:
                                ttft = float(
                                    clean_cells[header_map["ttft"]].split()[0])
                            except ValueError:
                                pass
                        if "itl" in header_map and clean_cells[
                                header_map["itl"]]:
                            try:
                                itl = float(
                                    clean_cells[header_map["itl"]].split()[0])
                            except ValueError:
                                pass
                        runs.append(
                            BenchmarkRunMetrics(
                                regime=row_regime,
                                concurrency=c_val,
                                output_tps=tps_val,
                                ttft_p50=ttft,
                                itl_p50=itl,
                                source=filename,
                            ))
                except ValueError:
                    pass
    return runs


def load_runs_from_file(path: Path) -> List[BenchmarkRunMetrics]:
    """Loads benchmark runs from a single JSON, CSV, or Markdown file."""
    runs: List[BenchmarkRunMetrics] = []
    if not path.is_file():
        return runs

    inferred_regime, inferred_conc = infer_regime_and_conc_from_path(path)
    source_desc = path.name
    if path.parent and path.name in ("clean.json", "result.json"):
        source_desc = f"{path.parent.name}/{path.name}"

    if path.suffix == ".md":
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            runs.extend(parse_markdown_tables(content, source_desc))
        except Exception:
            pass
        return runs

    if path.suffix == ".csv":
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    r = BenchmarkRunMetrics.from_dict(
                        row,
                        default_regime=inferred_regime,
                        default_concurrency=inferred_conc,
                        source=source_desc,
                    )
                    if r and r.regime in REQUIRED_REGIMES:
                        runs.append(r)
        except Exception:
            pass
        return runs

    if path.suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8",
                                             errors="ignore"))
        except Exception:
            return runs

        def _extract(obj: Any, reg_hint: str,
                     conc_hint: Optional[int]) -> None:
            if isinstance(obj, list):
                for item in obj:
                    _extract(item, reg_hint, conc_hint)
            elif isinstance(obj, dict):
                # Check if dict has a points or results list
                for list_key in ("points", "results", "runs", "sweeps",
                                 "benchmarks"):
                    if isinstance(obj.get(list_key), list):
                        _extract(obj[list_key], reg_hint, conc_hint)
                # Check if dict is keyed by regime names
                for k, v in obj.items():
                    norm_k = normalize_regime_name(k)
                    if norm_k in REQUIRED_REGIMES and isinstance(
                            v, (list, dict)):
                        _extract(v, norm_k, conc_hint)
                r = BenchmarkRunMetrics.from_dict(
                    obj,
                    default_regime=reg_hint,
                    default_concurrency=conc_hint,
                    source=source_desc,
                )
                if r and r.regime in REQUIRED_REGIMES:
                    runs.append(r)

        _extract(data, inferred_regime, inferred_conc)
    return runs


def extract_runs_from_trajectory(traj_path: Path) -> List[BenchmarkRunMetrics]:
    """Fallback parser that extracts benchmark sweep measurements from trajectory.json."""
    runs: List[BenchmarkRunMetrics] = []
    if not traj_path or not traj_path.is_file():
        return runs
    try:
        traj = json.loads(
            traj_path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return runs

    for step in traj.get("steps") or []:
        if not isinstance(step, dict):
            continue
        if step.get("source") == "user":
            continue
        sid = step.get("step_id")
        if sid is not None and int(sid) <= 2:
            continue
        texts: List[str] = []
        if isinstance(step.get("message"), str):
            texts.append(step["message"])
        obs = step.get("observation")
        if isinstance(obs, dict):
            for res in obs.get("results") or []:
                if isinstance(res, dict):
                    c = res.get("content")
                    if isinstance(c, str):
                        texts.append(c)
                    elif isinstance(c, list):
                        for item in c:
                            if isinstance(item, dict) and isinstance(
                                    item.get("text"), str):
                                texts.append(item["text"])

        for txt in texts:
            if "|" in txt and ("tok/s" in txt.lower()
                               or "throughput" in txt.lower()):
                runs.extend(
                    parse_markdown_tables(txt, f"trajectory:step_{sid}"))
            # Also match explicit prose reports from yeet-vllm-bench / subagents
            # Line-by-line regime context so multi-regime summaries don't cross-pollute
            reg_hint = None
            for line in txt.splitlines():
                line_reg = None
                if any(k in line.lower()
                       for k in ("8k/1k", "8k1k", "8192/1024", "isl=8192",
                                 "isl 8192")):
                    line_reg = "8k/1k"
                elif any(k in line.lower()
                         for k in ("1k/8k", "1k8k", "1024/8192", " osl=8192",
                                   " osl 8192")):
                    line_reg = "1k/8k"
                elif any(k in line.lower()
                         for k in ("1k/1k", "1k1k", "1024/1024", "isl=1024")):
                    line_reg = "1k/1k"
                if line_reg:
                    reg_hint = line_reg
                if reg_hint:
                    for m in re.finditer(
                            r"([\d,]+(?:\.\d+)?)\s*(?:output\s+)?tok/s\s+at\s+c=(\d+)",
                            line, re.I):
                        tps = float(m.group(1).replace(",", ""))
                        c_val = int(m.group(2))
                        runs.append(
                            BenchmarkRunMetrics(
                                regime=reg_hint,
                                concurrency=c_val,
                                output_tps=tps,
                                source=f"trajectory:step_{sid}",
                            ))
                    for m in re.finditer(
                            r"c=(\d+)[^.\n]{0,40}?([\d,]+(?:\.\d+)?)\s*(?:output\s+)?tok/s",
                            line, re.I):
                        c_val = int(m.group(1))
                        tps = float(m.group(2).replace(",", ""))
                        runs.append(
                            BenchmarkRunMetrics(
                                regime=reg_hint,
                                concurrency=c_val,
                                output_tps=tps,
                                source=f"trajectory:step_{sid}",
                            ))
    return runs


def evaluate_regime_sweeps(workspace: Path) -> Dict[str, Any]:
    """Discovers and summarizes all required ISL/OSL regime sweeps from artifacts + trajectory."""
    search_roots: List[Path] = []
    logs_dir = Path(os.environ.get("LOGS_DIR", "/logs"))
    if (logs_dir / "artifacts").is_dir():
        search_roots.append(logs_dir / "artifacts")
    if Path("/logs/artifacts").is_dir() and Path(
            "/logs/artifacts") not in search_roots:
        search_roots.append(Path("/logs/artifacts"))
    if workspace and workspace.is_dir():
        if (workspace / "artifacts").is_dir():
            search_roots.append(workspace / "artifacts")
        search_roots.append(workspace)

    skip_names = {
        "trajectory.json", "reward.json", "reward-details.json", "config.json",
        "result.json", "lock.json", "manifest.json", "image_probe.json",
        "tp_plan.json", "model_resolution.json", "deployments.json",
        "verification.json", "health.json", "models.json", "request.json"
    }
    skip_dirs = {
        ".git", "agent", "verifier", "verifier_legacy_regex", "preflight",
        "profiles"
    }

    artifact_runs: List[BenchmarkRunMetrics] = []
    seen_files: set[Path] = set()
    for root in search_roots:
        for subfile in sorted(root.rglob("*")):
            if not subfile.is_file() or subfile.suffix not in (".json", ".csv",
                                                               ".md"):
                continue
            resolved = subfile.resolve()
            if resolved in seen_files or subfile.name in skip_names:
                continue
            if any(p in skip_dirs for p in subfile.parts):
                continue
            seen_files.add(resolved)
            artifact_runs.extend(load_runs_from_file(subfile))

    by_regime: Dict[str, Dict[int, BenchmarkRunMetrics]] = {
        r: {}
        for r in REQUIRED_REGIMES
    }
    for r in artifact_runs:
        if r.regime in by_regime:
            existing = by_regime[r.regime].get(r.concurrency)
            if not existing:
                by_regime[r.regime][r.concurrency] = r
            else:
                existing_is_json = existing.source.endswith(".json")
                new_is_json = r.source.endswith(".json")
                if existing_is_json and not new_is_json:
                    if r.output_tps > existing.output_tps + 1.0:
                        by_regime[r.regime][r.concurrency] = r
                elif new_is_json and not existing_is_json:
                    if r.output_tps >= existing.output_tps - 1.0:
                        by_regime[r.regime][r.concurrency] = r
                elif r.output_tps > existing.output_tps:
                    by_regime[r.regime][r.concurrency] = r

    required_for_task = detect_required_regimes(workspace)
    missing_required = [reg for reg in required_for_task if not by_regime[reg]]

    # Fallback to trajectory ONLY for required regimes that are completely missing from artifacts
    if missing_required:
        traj_candidates = []
        if os.environ.get("ATIF_TRAJECTORY"):
            traj_candidates.append(Path(os.environ["ATIF_TRAJECTORY"]))
        if os.environ.get("TRAJECTORY_PATH"):
            traj_candidates.append(Path(os.environ["TRAJECTORY_PATH"]))
        if workspace:
            traj_candidates.extend([
                workspace / "agent" / "trajectory.json",
                workspace / "trajectory.json",
            ])
        traj_candidates.append(Path("/logs/agent/trajectory.json"))
        for tc in traj_candidates:
            if tc and tc.is_file():
                traj_runs = extract_runs_from_trajectory(tc)
                for r in traj_runs:
                    if r.regime in missing_required:
                        existing = by_regime[r.regime].get(r.concurrency)
                        if not existing or r.output_tps > existing.output_tps:
                            by_regime[r.regime][r.concurrency] = r
                break

    regimes_summary: Dict[str, Any] = {}
    covered_count = 0
    required_covered_count = 0
    for reg in REQUIRED_REGIMES:
        cmap = by_regime[reg]
        if not cmap:
            regimes_summary[reg] = {
                "covered": False,
                "required": reg in required_for_task,
                "num_points": 0,
                "peak_tps": 0.0,
                "peak_concurrency": None,
                "peak_source": None,
                "points": [],
            }
            continue
        covered_count += 1
        if reg in required_for_task:
            required_covered_count += 1
        pts = [cmap[c] for c in sorted(cmap.keys())]
        peak = max(pts, key=lambda x: x.output_tps)
        regimes_summary[reg] = {
            "covered":
            True,
            "required":
            reg in required_for_task,
            "num_points":
            len(pts),
            "peak_tps":
            round(peak.output_tps, 2),
            "peak_concurrency":
            peak.concurrency,
            "peak_source":
            peak.source,
            "points": [{
                "concurrency": p.concurrency,
                "output_tps": round(p.output_tps, 2),
                "ttft_p50_ms": p.ttft_p50,
                "itl_p50_ms": p.itl_p50,
                "source": p.source,
            } for p in pts],
        }

    coverage = round(required_covered_count / len(required_for_task), 4)
    return {
        "regime_coverage": coverage,
        "covered_regimes": required_covered_count,
        "required_regimes": len(required_for_task),
        "peak_tps_1k_1k": regimes_summary["1k/1k"]["peak_tps"],
        "peak_tps_1k_8k": regimes_summary["1k/8k"]["peak_tps"],
        "peak_tps_8k_1k": regimes_summary["8k/1k"]["peak_tps"],
        "regimes": regimes_summary,
    }


def format_regime_trace(summary: Dict[str, Any]) -> str:
    """Formats an ASCII debug trace of all discovered regime sweep points."""
    req_cnt = summary.get("required_regimes", 3)
    lines = [
        "=" * 80,
        "DETERMINISTIC ONBOARDING REGIME SWEEP TRACE (1k/1k, 1k/8k, 8k/1k)",
        "=" * 80,
        f"Regime Coverage : {summary['covered_regimes']}/{req_cnt} ({summary['regime_coverage']:.4f})",
        f"Peak 1k/1k TPS  : {summary['peak_tps_1k_1k']:,.2f} tok/s",
        f"Peak 1k/8k TPS  : {summary['peak_tps_1k_8k']:,.2f} tok/s",
        f"Peak 8k/1k TPS  : {summary['peak_tps_8k_1k']:,.2f} tok/s",
        "-" * 80,
    ]
    for reg in REQUIRED_REGIMES:
        rdata = summary["regimes"][reg]
        if not rdata["covered"]:
            lines.append(f"[REGIME: {reg}] -> NOT FOUND")
            lines.append("-" * 80)
            continue
        lines.append(
            f"[REGIME: {reg}] -> Peak: {rdata['peak_tps']:,.2f} tok/s @ c={rdata['peak_concurrency']} "
            f"({rdata['num_points']} concurrency points, source={rdata['peak_source']})"
        )
        lines.append(
            f"  {'Concurrency':<12} | {'Output TPS':<14} | {'TTFT (ms)':<12} | {'ITL (ms)':<12} | {'Source'}"
        )
        lines.append(f"  {'-'*12}-+-{'-'*14}-+-{'-'*12}-+-{'-'*12}-+-{'-'*24}")
        for pt in rdata["points"]:
            c_str = f"c={pt['concurrency']}"
            tps_str = f"{pt['output_tps']:,.2f}"
            ttft_str = f"{pt['ttft_p50_ms']:.1f}" if pt[
                "ttft_p50_ms"] is not None else "—"
            itl_str = f"{pt['itl_p50_ms']:.2f}" if pt[
                "itl_p50_ms"] is not None else "—"
            marker = " <-- PEAK" if pt["concurrency"] == rdata[
                "peak_concurrency"] else ""
            lines.append(
                f"  {c_str:<12} | {tps_str:<14} | {ttft_str:<12} | {itl_str:<12} | {pt['source']}{marker}"
            )
        lines.append("-" * 80)
    return "\n".join(lines)


@criterion(description=(
    "Deterministically verifies and extracts benchmark concurrency sweeps across "
    "the three required ISL/OSL regimes (1k/1k, 1k/8k, 8k/1k) from artifacts and trajectory."
))
def regime_sweep_coverage(workspace: Path) -> float:
    """Calculates regime sweep coverage and prints per-regime peak TPS trace."""
    summary = evaluate_regime_sweeps(workspace)
    print(format_regime_trace(summary))
    # Save structured summary for test.sh to merge into reward.json
    logs_dir = Path(os.environ.get("LOGS_DIR", "/logs"))
    verifier_dir = logs_dir / "verifier" if (logs_dir /
                                             "verifier").is_dir() else logs_dir
    try:
        verifier_dir.mkdir(parents=True, exist_ok=True)
        (verifier_dir / "regime_sweep_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
    except OSError:
        pass
    return float(summary["regime_coverage"])
