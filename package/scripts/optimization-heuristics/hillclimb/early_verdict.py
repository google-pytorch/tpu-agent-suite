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
"""Equal-time early stopping from `aiperf` realtime telemetry blocks.

Closed-loop multi-turn benchmarks emit cumulative telemetry blocks during the
measured phase. Early in a full-hour run the cumulative rate underestimates the
final tok/s/chip by 10-15%, but runs of the same trace share that bias, so a
running candidate's cumulative tok/s/chip over a completed baseline's at the
*same elapsed measured-phase second* predicts the final ratio. On 15 finished
full-hour GLM-5.2 P/D runs (105 pairs), the 15-minute ratio was within a median
`1.6%` (max `5.7%`) of the final ratio for the 31 pairs with the same
concurrency and chip count, and a median `5.3%` (max `14.1%`) for the other 74.
Stopping a losing arm at minute 10 frees its chips about 50 minutes early.
"""

import dataclasses
import re
from collections.abc import Sequence

_TIME_RE = re.compile(r"(\d+):(\d{2}):(\d{2})")
_CELL_RE = re.compile(r"([\d,]+(?:\.\d+)?)")
_PHASE_RE = re.compile(r"Profiling\s+([\d,]+)/([\d,]+)")


@dataclasses.dataclass(frozen=True)
class Block:
    """One `aiperf` realtime telemetry block from the measured phase."""

    completed: int
    target: int
    elapsed_s: int
    req_per_s: float
    output_tok_per_s: float
    total_tok_per_s: float
    ttft_ms: float | None = None
    itl_ms: float | None = None
    seq_s: float | None = None


@dataclasses.dataclass(frozen=True)
class Band:
    """Calibrated error band from `min_elapsed_s` onward."""

    min_elapsed_s: int
    rel_error: float


# The 10-minute and 30-minute bands (cross shape: 20 and 30 minutes) are chosen
# margins, not measured errors; the error was measured at 15 minutes only. The
# 10-minute band is wider than the largest same-shape 15-minute error (5.7%).
# The later bands are narrower than the 15-minute maximum and have no measured
# error behind them.

# Same concurrency and chip count (31 calibration pairs; max error 5.7% at
# 15 min).
SAME_SHAPE_BANDS: tuple[Band, ...] = (
    Band(1800, 0.04),
    Band(900, 0.06),
    Band(600, 0.08),
)

# Different concurrency or chip count (74 calibration pairs; median error 5.3%,
# max 14.1% at 15 min).
CROSS_SHAPE_BANDS: tuple[Band, ...] = (
    Band(1800, 0.07),
    Band(1200, 0.09),
    Band(900, 0.11),
)

ITL_SLOWDOWN = 1.5

TOO_EARLY = "too_early"
STOP = "stop"
PROMISING = "promising"
WAIT = "wait"


def _num(cell: str) -> float | None:
    match = _CELL_RE.search(cell)
    return float(match.group(1).replace(",", "")) if match else None


def _seconds(cell: str) -> int | None:
    match = _TIME_RE.search(cell)
    if not match:
        return None
    hours, mins, secs = (int(g) for g in match.groups())
    return hours * 3600 + mins * 60 + secs


def parse_blocks(text: str) -> list[Block]:
    """Extracts measured-phase `Realtime Telemetry` blocks from an `aiperf` log."""
    blocks: list[Block] = []
    phase: tuple[int, int] | None = None
    elapsed_s: int | None = None
    metrics: dict[str, float] = {}

    def flush() -> None:
        nonlocal phase, elapsed_s, metrics
        if phase and elapsed_s and "Total" in metrics:
            blocks.append(
                Block(
                    completed=phase[0],
                    target=phase[1],
                    elapsed_s=elapsed_s,
                    req_per_s=metrics.get("Requests", 0.0),
                    output_tok_per_s=metrics.get("Output", 0.0),
                    total_tok_per_s=metrics["Total"],
                    ttft_ms=metrics.get("TTFT"),
                    itl_ms=metrics.get("ITL"),
                    seq_s=metrics.get("Seq"),
                ))
        phase, elapsed_s, metrics = None, None, {}

    for line in text.splitlines():
        if "Realtime Telemetry" in line:
            flush()
            continue
        cells = [c.strip() for c in line.split("│")]
        if len(cells) >= 5:
            match = _PHASE_RE.search(cells[1])
            if match:
                phase = (
                    int(match.group(1).replace(",", "")),
                    int(match.group(2).replace(",", "")),
                )
                elapsed_s = _seconds(cells[3])
                continue
        if phase is not None and len(cells) >= 8:
            name = cells[1]
            val = _num(cells[2])
            if name and val is not None:
                metrics[name] = val
    flush()
    return blocks


def block_at(blocks: Sequence[Block], elapsed_s: int) -> Block | None:
    """Returns the block closest in elapsed seconds to `elapsed_s` (within 60s)."""
    if not blocks:
        return None
    closest = min(blocks, key=lambda b: abs(b.elapsed_s - elapsed_s))
    if abs(closest.elapsed_s - elapsed_s) > 60:
        return None
    return closest


def band_at(elapsed_s: int, bands: Sequence[Band]) -> float | None:
    """Returns the calibrated relative error band at `elapsed_s`."""
    applicable = [b.rel_error for b in bands if elapsed_s >= b.min_elapsed_s]
    return min(applicable) if applicable else None


@dataclasses.dataclass(frozen=True)
class Verdict:
    """Equal-time early-stopping assessment."""

    status: str  # too_early | stop | promising | wait
    reason: str
    elapsed_s: int = 0
    arm_tok_s_chip: float | None = None
    base_tok_s_chip: float | None = None
    ratio: float | None = None
    band: float | None = None
    projected_low: float | None = None
    projected_high: float | None = None
    arm_itl_ms: float | None = None
    base_itl_ms: float | None = None
    target_ratio: float = 1.0


def assess(
    arm_blocks: Sequence[Block],
    arm_chips: int,
    base_blocks: Sequence[Block],
    base_chips: int,
    *,
    same_shape: bool = True,
    target_ratio: float = 1.0,
    itl_slowdown: float = ITL_SLOWDOWN,
) -> Verdict:
    """Compares a running arm's `aiperf` log against a finished baseline's log."""
    if arm_chips <= 0 or base_chips <= 0:
        raise ValueError("arm_chips and base_chips must be positive")
    if not arm_blocks:
        return Verdict(
            status=TOO_EARLY,
            reason="Measured phase has not started yet (warmup or compiling).",
            target_ratio=target_ratio,
        )
    now = arm_blocks[-1]
    base_now = block_at(base_blocks, now.elapsed_s)
    if base_now is None or base_now.total_tok_per_s <= 0:
        return Verdict(
            status=TOO_EARLY,
            reason=(
                f"Baseline has no telemetry block near {now.elapsed_s}s to "
                "compare against."),
            elapsed_s=now.elapsed_s,
            target_ratio=target_ratio,
        )

    arm_tpc = now.total_tok_per_s / arm_chips
    base_tpc = base_now.total_tok_per_s / base_chips
    ratio = arm_tpc / base_tpc
    bands = SAME_SHAPE_BANDS if same_shape else CROSS_SHAPE_BANDS
    band = band_at(now.elapsed_s, bands)

    itl_broken = (now.itl_ms is not None and base_now.itl_ms is not None
                  and base_now.itl_ms > 0
                  and now.itl_ms >= base_now.itl_ms * itl_slowdown)

    if band is None:
        earliest = min(b.min_elapsed_s for b in bands)
        if now.elapsed_s >= 300 and itl_broken and ratio < target_ratio:
            return Verdict(
                status=STOP,
                reason=(
                    f"At {now.elapsed_s / 60:.1f} min, ITL is "
                    f"{now.itl_ms:.1f} ms vs {base_now.itl_ms:.1f} ms "
                    f"({now.itl_ms / base_now.itl_ms:.2f}x >= {itl_slowdown}x) "
                    f"and equal-time tok/s/chip is {ratio:.3f}x."),
                elapsed_s=now.elapsed_s,
                arm_tok_s_chip=arm_tpc,
                base_tok_s_chip=base_tpc,
                ratio=ratio,
                arm_itl_ms=now.itl_ms,
                base_itl_ms=base_now.itl_ms,
                target_ratio=target_ratio,
            )
        return Verdict(
            status=TOO_EARLY,
            reason=(
                f"{now.elapsed_s}s into measured phase; calibrated equal-time "
                f"band starts at {earliest}s ({earliest // 60} min)."),
            elapsed_s=now.elapsed_s,
            arm_tok_s_chip=arm_tpc,
            base_tok_s_chip=base_tpc,
            ratio=ratio,
            arm_itl_ms=now.itl_ms,
            base_itl_ms=base_now.itl_ms,
            target_ratio=target_ratio,
        )

    low, high = ratio - band, ratio + band
    mins = now.elapsed_s / 60.0
    if high < target_ratio:
        status = STOP
        reason = (
            f"At {mins:.1f} min, equal-time tok/s/chip ratio is {ratio:.3f} "
            f"(band [{low:.3f}, {high:.3f}]), below target {target_ratio:.3f}."
        )
    elif itl_broken and ratio < target_ratio:
        status = STOP
        reason = (
            f"At {mins:.1f} min, ITL is {now.itl_ms:.1f} ms vs "
            f"{base_now.itl_ms:.1f} ms ({now.itl_ms / base_now.itl_ms:.2f}x) "
            f"and equal-time ratio {ratio:.3f} < {target_ratio:.3f}.")
    elif low > target_ratio and not itl_broken:
        status = PROMISING
        reason = (
            f"At {mins:.1f} min, equal-time tok/s/chip ratio is {ratio:.3f} "
            f"(band [{low:.3f}, {high:.3f}]), above target {target_ratio:.3f}; "
            "let it finish.")
    else:
        status = WAIT
        reason = (
            f"At {mins:.1f} min, equal-time tok/s/chip ratio is {ratio:.3f} "
            f"(band [{low:.3f}, {high:.3f}] straddles target "
            f"{target_ratio:.3f}); check again later.")
    return Verdict(
        status=status,
        reason=reason,
        elapsed_s=now.elapsed_s,
        arm_tok_s_chip=arm_tpc,
        base_tok_s_chip=base_tpc,
        ratio=ratio,
        band=band,
        projected_low=low,
        projected_high=high,
        arm_itl_ms=now.itl_ms,
        base_itl_ms=base_now.itl_ms,
        target_ratio=target_ratio,
    )


def render(verdict: Verdict,
           arm: str = "candidate",
           baseline: str = "baseline") -> str:
    """Formats an early-stopping `Verdict` as readable text."""
    lines = [
        f"early_verdict: {verdict.status.upper()} ({arm} vs {baseline})",
        f"  {verdict.reason}",
    ]
    if verdict.arm_tok_s_chip is not None and verdict.base_tok_s_chip is not None:
        lines.append(f"  tok/s/chip at {verdict.elapsed_s}s: {arm} "
                     f"{verdict.arm_tok_s_chip:.1f} vs {baseline} "
                     f"{verdict.base_tok_s_chip:.1f}")
    if verdict.arm_itl_ms is not None and verdict.base_itl_ms is not None:
        lines.append(f"  ITL mean at {verdict.elapsed_s}s: {arm} "
                     f"{verdict.arm_itl_ms:.1f} ms vs {baseline} "
                     f"{verdict.base_itl_ms:.1f} ms")
    return "\n".join(lines) + "\n"
