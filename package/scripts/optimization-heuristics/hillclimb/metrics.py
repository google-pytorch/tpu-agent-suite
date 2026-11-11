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
"""Cost-efficiency math (`mtok_per_usd`) and benchmark metrics.

This package ships no accelerator prices: `Mtok/$` needs the caller's own
chip-hour rate. Relative comparisons use tok/s/chip, which gives the same
percentage change as `Mtok/$` when both runs pay the same rate.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any

SECONDS_PER_HOUR = 3600.0
TOKENS_PER_MILLION = 1_000_000.0


def mtok_per_usd(
    token_throughput_per_chip: float,
    chip_hour_cost_usd: float,
) -> float:
    """Computes million total tokens per dollar (`Mtok/$`).

    Formula:
      `(token_throughput_per_chip * 3600 / chip_hour_cost_usd) / 1e6`

    Args:
      token_throughput_per_chip: Total input + output tokens per second per chip.
      chip_hour_cost_usd: The caller's dollar cost of one accelerator
        chip-hour. There is no default.

    Returns:
      Million tokens served per dollar.

    Raises:
      ValueError: If `chip_hour_cost_usd <= 0` or throughput is negative.
    """
    if chip_hour_cost_usd <= 0:
        raise ValueError(
            f"chip_hour_cost_usd must be positive, got {chip_hour_cost_usd}")
    if token_throughput_per_chip < 0:
        raise ValueError("token_throughput_per_chip must be non-negative, got "
                         f"{token_throughput_per_chip}")
    tokens_per_chip_hour = token_throughput_per_chip * SECONDS_PER_HOUR
    return (tokens_per_chip_hour / chip_hour_cost_usd) / TOKENS_PER_MILLION


def _opt_float(m: Mapping[str, Any], *keys: str) -> float | None:
    for key in keys:
        val = m.get(key)
        if not isinstance(val, bool) and isinstance(val, (int, float)):
            return float(val)
    return None


def _opt_int(m: Mapping[str, Any], *keys: str) -> int | None:
    val = _opt_float(m, *keys)
    return int(val) if val is not None else None


@dataclasses.dataclass(frozen=True)
class ServingMetrics:
    """Parsed benchmark metrics for a single serving run."""

    run_id: str
    token_throughput_per_chip: float
    p90_interactivity_tok_s: float = 0.0
    mean_ttft_ms: float = 0.0
    p90_ttft_ms: float = 0.0
    p99_ttft_ms: float = 0.0
    median_itl_ms: float = 0.0
    p90_itl_ms: float = 0.0
    p99_itl_ms: float = 0.0
    mean_e2el_ms: float = 0.0
    p90_e2el_ms: float = 0.0
    request_throughput: float = 0.0
    mean_input_tokens: float = 0.0
    mean_output_tokens: float = 0.0
    kv_cache_hit_rate: float | None = None
    spec_decode_mean_acceptance_length: float | None = None
    num_requests_total: int = 0
    records_error_dropped: int = 0

    def mtok_per_dollar(self, chip_hour_cost_usd: float) -> float:
        """Returns cost-efficiency (`Mtok/$`) at the caller's chip-hour rate."""
        return mtok_per_usd(self.token_throughput_per_chip, chip_hour_cost_usd)

    @classmethod
    def from_dict(
        cls,
        raw: Mapping[str, Any],
        run_id: str = "",
    ) -> "ServingMetrics":
        """Extracts `ServingMetrics` from a benchmark summary or metrics map."""
        m_raw = raw.get("metrics")
        m: Mapping[str, Any] = m_raw if isinstance(m_raw, Mapping) else raw
        resolved_id = (run_id or str(raw.get("run_id") or raw.get("id") or ""))
        tpc = _opt_float(m, "token_throughput_per_chip")
        if tpc is None:
            raise ValueError(
                f"Run {resolved_id!r} has no token_throughput_per_chip metric."
            )
        return cls(
            run_id=resolved_id,
            token_throughput_per_chip=tpc,
            p90_interactivity_tok_s=(_opt_float(m, "p90_interactivity_tok_s",
                                                "p90_interactivity") or 0.0),
            mean_ttft_ms=_opt_float(m, "mean_ttft_ms") or 0.0,
            p90_ttft_ms=_opt_float(m, "p90_ttft_ms") or 0.0,
            p99_ttft_ms=_opt_float(m, "p99_ttft_ms") or 0.0,
            median_itl_ms=(_opt_float(m, "median_itl_ms", "median_tpot_ms")
                           or 0.0),
            p90_itl_ms=_opt_float(m, "p90_itl_ms", "p90_tpot_ms") or 0.0,
            p99_itl_ms=_opt_float(m, "p99_itl_ms", "p99_tpot_ms") or 0.0,
            mean_e2el_ms=_opt_float(m, "mean_e2el_ms") or 0.0,
            p90_e2el_ms=_opt_float(m, "p90_e2el_ms") or 0.0,
            request_throughput=_opt_float(m, "request_throughput") or 0.0,
            mean_input_tokens=_opt_float(m, "mean_input_tokens") or 0.0,
            mean_output_tokens=_opt_float(m, "mean_output_tokens") or 0.0,
            kv_cache_hit_rate=_opt_float(m, "kv_cache_hit_rate"),
            spec_decode_mean_acceptance_length=_opt_float(
                m, "spec_decode_mean_acceptance_length"),
            num_requests_total=(_opt_int(m, "num_requests_total",
                                         "num_requests_successful") or 0),
            records_error_dropped=(_opt_int(m, "records_error_dropped",
                                            "failed") or 0),
        )


@dataclasses.dataclass(frozen=True)
class MetricDelta:
    """Relative comparison between a baseline and candidate serving run.

    The percentage changes need no price. The absolute `Mtok/$` values are set
    only when `compare_runs` is given a chip-hour rate.
    """

    tok_s_chip_pct_change: float
    p90_interactivity_pct_change: float
    mean_ttft_pct_change: float
    median_itl_pct_change: float
    baseline_mtok_per_usd: float | None = None
    candidate_mtok_per_usd: float | None = None

    @property
    def mtok_per_usd_pct_change(self) -> float:
        """`Mtok/$` change, equal to the tok/s/chip change at one rate."""
        return self.tok_s_chip_pct_change


def _pct_change(before: float, after: float) -> float:
    if before == 0:
        return 0.0
    return ((after - before) / before) * 100.0


def compare_runs(
    baseline: ServingMetrics,
    candidate: ServingMetrics,
    chip_hour_cost_usd: float | None = None,
) -> MetricDelta:
    """Computes relative KPI deltas of `candidate` vs `baseline`.

    `chip_hour_cost_usd` only adds each run's absolute `Mtok/$`; the deltas
    are the same with or without it.
    """
    base_mtok: float | None = None
    cand_mtok: float | None = None
    if chip_hour_cost_usd is not None:
        base_mtok = baseline.mtok_per_dollar(chip_hour_cost_usd)
        cand_mtok = candidate.mtok_per_dollar(chip_hour_cost_usd)
    return MetricDelta(
        tok_s_chip_pct_change=_pct_change(baseline.token_throughput_per_chip,
                                          candidate.token_throughput_per_chip),
        baseline_mtok_per_usd=base_mtok,
        candidate_mtok_per_usd=cand_mtok,
        p90_interactivity_pct_change=_pct_change(
            baseline.p90_interactivity_tok_s,
            candidate.p90_interactivity_tok_s),
        mean_ttft_pct_change=_pct_change(baseline.mean_ttft_ms,
                                         candidate.mean_ttft_ms),
        median_itl_pct_change=_pct_change(baseline.median_itl_ms,
                                          candidate.median_itl_ms),
    )
