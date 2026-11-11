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
"""Prefix-cache HBM + CPU offload tier KV replay and Little's Law lane saturation.

Models two coupled serving phenomena on multi-turn long-context traffic:
  1. Prefix-cache replay (`replay`): block-aligned HBM LRU + optional CPU host
     offload tier, including the EAGLE/MTP unannotated last-chunk pop hazard
     (`tier_pop=True` vs `TPU_OFFLOAD_KEEP_EAGLE_TAIL=1`) and DCP-scaled block
     counts (`PREFILL_DCP_SIZE=8` halving HBM prefix capacity vs `16`).
  2. Little's Law lane decomposition (`littles_law_seconds` & `lane_saturation`):
     partitions `C` closed-loop client lanes into prefill, decode admission
     wait, KV handoff pull, active decode generation, and client think/idle
     lanes to expose whether decode slots are saturated or starved upstream.
"""

import collections
import dataclasses
import json
import pathlib
import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import Any


@dataclasses.dataclass(frozen=True)
class ReplayRequest:
    """One multi-turn serving request for KV prefix-cache replay."""

    isl: int
    block_hashes: Sequence[int | str]
    osl: int = 0
    phase: str = "profiling"
    start: float | None = None
    end: float | None = None
    prefill_s: float | None = None
    admit_wait_s: float | None = None
    kv_pull_s: float | None = None
    decode_s: float | None = None


@dataclasses.dataclass(frozen=True)
class CacheConfig:
    """Geometry and eviction policy of the HBM + CPU-tier prefix cache."""

    unit_tokens: int = 64
    units_per_block: int = 32
    gpu_blocks: int = 5806
    cpu_blocks: int = 3816
    tier_pop: bool = True
    infinite: bool = False

    @property
    def chunk_tokens(self) -> int:
        return self.unit_tokens * self.units_per_block

    @property
    def hbm_capacity_tokens(self) -> int:
        return self.gpu_blocks * self.chunk_tokens


@dataclasses.dataclass(frozen=True)
class Outcome:
    """Per-request KV cache hit breakdown."""

    hbm_tokens: int
    tier_tokens: int
    computed_tokens: int
    tier_chunks: int

    @property
    def tier_served(self) -> bool:
        return self.tier_chunks > 0


def _move_or_insert(
    lru: collections.OrderedDict[int | str, None],
    key: int | str,
    cap: int,
) -> int | str | None:
    if cap <= 0:
        return None
    if key in lru:
        lru.move_to_end(key)
        return None
    evicted: int | str | None = None
    if len(lru) >= cap:
        evicted, _ = lru.popitem(last=False)
    lru[key] = None
    return evicted


def replay(
    requests: Sequence[ReplayRequest],
    config: CacheConfig,
) -> list[Outcome | None]:
    """Replays `requests` through the two-tier HBM + CPU prefix cache."""
    u = config.unit_tokens
    upb = config.units_per_block
    chunk_toks = config.chunk_tokens
    hbm: collections.OrderedDict[int | str, None] = collections.OrderedDict()
    cpu: collections.OrderedDict[int | str, None] = collections.OrderedDict()
    seen: set[int | str] = set()
    outcomes: list[Outcome | None] = []

    for req in requests:
        if req.isl <= 0 or not req.block_hashes:
            outcomes.append(None)
            continue

        if config.infinite:
            n_units = min(req.isl // u, len(req.block_hashes))
            hit_units = 0
            for idx in range(n_units):
                if req.block_hashes[idx] in seen:
                    hit_units += 1
                else:
                    break
            if hit_units * u >= req.isl and hit_units > 0:
                hit_units -= 1
            for idx in range(n_units):
                seen.add(req.block_hashes[idx])
            hit_toks = hit_units * u
            outcomes.append(
                Outcome(
                    hbm_tokens=hit_toks,
                    tier_tokens=0,
                    computed_tokens=req.isl - hit_toks,
                    tier_chunks=0,
                ))
            continue

        n_chunks = min(req.isl // chunk_toks, len(req.block_hashes) // upb)
        chunk_keys = [
            req.block_hashes[(c + 1) * upb - 1] for c in range(n_chunks)
        ]

        hbm_chunks = 0
        for key in chunk_keys:
            if key in hbm:
                hbm_chunks += 1
            else:
                break
        if hbm_chunks * chunk_toks >= req.isl and hbm_chunks > 0:
            hbm_chunks -= 1

        tier_chunks = 0
        if config.cpu_blocks > 0:
            for c in range(hbm_chunks, n_chunks):
                if chunk_keys[c] in cpu:
                    tier_chunks += 1
                else:
                    break
            if config.tier_pop and tier_chunks > 0:
                tier_chunks -= 1
            if (hbm_chunks +
                    tier_chunks) * chunk_toks >= req.isl and tier_chunks > 0:
                tier_chunks -= 1

        for c in range(hbm_chunks):
            hbm.move_to_end(chunk_keys[c])
        for c in range(hbm_chunks, hbm_chunks + tier_chunks):
            key = chunk_keys[c]
            cpu.move_to_end(key)
            _move_or_insert(hbm, key, config.gpu_blocks)
        for c in range(hbm_chunks + tier_chunks, n_chunks):
            key = chunk_keys[c]
            _move_or_insert(hbm, key, config.gpu_blocks)
            if config.cpu_blocks > 0:
                _move_or_insert(cpu, key, config.cpu_blocks)

        hbm_toks = hbm_chunks * chunk_toks
        tier_toks = tier_chunks * chunk_toks
        computed_toks = max(0, req.isl - hbm_toks - tier_toks)
        outcomes.append(
            Outcome(
                hbm_tokens=hbm_toks,
                tier_tokens=tier_toks,
                computed_tokens=computed_toks,
                tier_chunks=tier_chunks,
            ))
    return outcomes


def totals(
    requests: Sequence[ReplayRequest],
    outcomes: Sequence[Outcome | None],
    phase: str | None = "profiling",
) -> dict[str, float | int]:
    """Aggregates token and hit shares across `requests`."""
    total_tok = 0
    hbm_tok = 0
    tier_tok = 0
    comp_tok = 0
    checked = 0
    tier_reqs = 0
    for req, out in zip(requests, outcomes):
        if out is None or (phase is not None and req.phase != phase):
            continue
        checked += 1
        total_tok += req.isl
        hbm_tok += out.hbm_tokens
        tier_tok += out.tier_tokens
        comp_tok += out.computed_tokens
        if out.tier_served:
            tier_reqs += 1
    denom = float(total_tok) if total_tok > 0 else 1.0
    return {
        "requests": checked,
        "total_tokens": total_tok,
        "hbm_tokens": hbm_tok,
        "tier_tokens": tier_tok,
        "computed_tokens": comp_tok,
        "hbm_share": hbm_tok / denom,
        "tier_share": tier_tok / denom,
        "hit_share": (hbm_tok + tier_tok) / denom,
        "computed_share": comp_tok / denom,
        "tier_served_requests": tier_reqs,
    }


@dataclasses.dataclass(frozen=True)
class CounterCheck:
    """Agreement between replay model and scraped vLLM prefix-cache counters."""

    checked: int
    within: int
    tier_served_checked: int
    tier_served_within: int


def check_against_counters(
    requests: Sequence[ReplayRequest],
    outcomes: Sequence[Outcome | None],
    slices: Sequence[tuple[float, float, float, float]],
    tolerance_tokens: int = 256,
) -> CounterCheck:
    """Checks replay hit counts against `(t0, t1, queries_delta, hits_delta)`."""
    checked = within = tier_checked = tier_within = 0
    for t0, t1, q_delta, h_delta in slices:
        matched = [(r, o) for r, o in zip(requests, outcomes)
                   if o is not None and r.start is not None
                   and r.end is not None and r.start >= t0 and r.end <= t1]
        if len(matched) != 1:
            continue
        req, out = matched[0]
        if abs(req.isl - q_delta) > tolerance_tokens:
            continue
        checked += 1
        pred_hit = out.hbm_tokens + out.tier_tokens
        is_ok = abs(pred_hit - h_delta) <= tolerance_tokens
        if is_ok:
            within += 1
        if out.tier_served:
            tier_checked += 1
            if is_ok:
                tier_within += 1
    return CounterCheck(checked, within, tier_checked, tier_within)


@dataclasses.dataclass(frozen=True)
class WaitRow:
    """Prefill turnaround grouped by hit class and arrival queue depth."""

    label: str
    depth_bucket: tuple[float, float]
    n: int
    mean_s: float
    p50_s: float
    p90_s: float


def wait_by_queue_depth(
    requests: Sequence[ReplayRequest],
    outcomes: Sequence[Outcome | None],
    depth_at: Callable[[float], float | None],
    buckets: Sequence[tuple[float, float]] = (
        (0.0, 0.0),
        (1.0, 2.0),
        (3.0, float("inf")),
    ),
) -> list[WaitRow]:
    """Stratifies prefill latency by hit class and queue depth at arrival."""
    grouped: dict[tuple[str, tuple[float, float]], list[float]] = {}
    for req, out in zip(requests, outcomes):
        if out is None or req.start is None or req.prefill_s is None:
            continue
        depth = depth_at(req.start)
        if depth is None:
            continue
        cls = "tier-served" if out.tier_served else "HBM-only"
        for low, high in buckets:
            if low <= depth <= high:
                grouped.setdefault((cls, (low, high)),
                                   []).append(req.prefill_s)
                break

    rows: list[WaitRow] = []
    for (cls, bkt), vals in sorted(grouped.items()):
        svals = sorted(vals)
        p90_idx = min(len(svals) - 1, int(round(0.90 * (len(svals) - 1))))
        rows.append(
            WaitRow(
                label=cls,
                depth_bucket=bkt,
                n=len(svals),
                mean_s=statistics.fmean(svals),
                p50_s=statistics.median(svals),
                p90_s=svals[p90_idx],
            ))
    return rows


def littles_law_seconds(
    mean_waiting: float,
    arrivals: int,
    duration_s: float,
) -> float:
    """Computes mean residence seconds `W = L / lambda` (`lambda = N / T`)."""
    if arrivals <= 0 or duration_s <= 0 or mean_waiting < 0:
        return 0.0
    arrival_rate = arrivals / duration_s
    return mean_waiting / arrival_rate


@dataclasses.dataclass(frozen=True)
class LaneSaturation:
    """Little's Law decomposition of closed-loop client concurrency lanes `C`."""

    client_concurrency: int
    decode_batch_cap: int
    throughput_req_s: float
    prefill_lanes: float
    admission_wait_lanes: float
    kv_pull_lanes: float
    decode_gen_lanes: float
    client_think_gap_lanes: float
    decode_fill_ratio: float
    regime: str  # decode_bound | decode_underfilled

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def lane_saturation(
    *,
    client_concurrency: int,
    decode_batch_cap: int,
    throughput_req_s: float,
    mean_prefill_s: float,
    mean_decode_s: float,
    mean_admit_wait_s: float = 0.0,
    mean_kv_pull_s: float = 0.0,
) -> LaneSaturation:
    """Applies `L_stage = X * W_stage` across P/D stages and client think gap."""
    if client_concurrency <= 0 or decode_batch_cap <= 0:
        raise ValueError("client_concurrency and decode_batch_cap must be > 0")
    x = max(0.0, throughput_req_s)
    prefill_l = x * max(0.0, mean_prefill_s)
    admit_l = x * max(0.0, mean_admit_wait_s)
    kv_l = x * max(0.0, mean_kv_pull_s)
    decode_l = x * max(0.0, mean_decode_s)
    server_l = prefill_l + admit_l + kv_l + decode_l
    think_l = max(0.0, float(client_concurrency) - server_l)
    fill_ratio = decode_l / float(decode_batch_cap)
    regime = ("decode_bound" if fill_ratio >= 0.85 or admit_l >= 1.5 else
              "decode_underfilled")
    return LaneSaturation(
        client_concurrency=client_concurrency,
        decode_batch_cap=decode_batch_cap,
        throughput_req_s=x,
        prefill_lanes=prefill_l,
        admission_wait_lanes=admit_l,
        kv_pull_lanes=kv_l,
        decode_gen_lanes=decode_l,
        client_think_gap_lanes=think_l,
        decode_fill_ratio=fill_ratio,
        regime=regime,
    )


def load_requests_jsonl(path: str | pathlib.Path) -> list[ReplayRequest]:
    """Loads `ReplayRequest` rows from a JSONL file."""
    out: list[ReplayRequest] = []
    with pathlib.Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if not isinstance(raw, Mapping):
                continue
            hashes = tuple(raw.get("block_hashes") or ())
            out.append(
                ReplayRequest(
                    isl=int(raw.get("isl",
                                    len(hashes) * 64)),
                    block_hashes=hashes,
                    osl=int(raw.get("osl", 0)),
                    phase=str(raw.get("phase", "profiling")),
                    start=(float(raw["start"])
                           if raw.get("start") is not None else None),
                    end=(float(raw["end"])
                         if raw.get("end") is not None else None),
                    prefill_s=(float(raw["prefill_s"])
                               if raw.get("prefill_s") is not None else None),
                    admit_wait_s=(float(raw["admit_wait_s"])
                                  if raw.get("admit_wait_s") is not None else
                                  None),
                    kv_pull_s=(float(raw["kv_pull_s"])
                               if raw.get("kv_pull_s") is not None else None),
                    decode_s=(float(raw["decode_s"])
                              if raw.get("decode_s") is not None else None),
                ))
    return out


def render_replay_markdown(
    policy_totals: Mapping[str, Mapping[str, float | int]],
    lanes: LaneSaturation | None = None,
) -> str:
    """Renders KV replay policy comparison and optional lane saturation table."""
    lines = [
        "### Prefix-Cache KV Replay",
        "",
        "| Policy | HBM hit | CPU tier | Total hit | Computed | Tier-served reqs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, t in policy_totals.items():
        comp_m = float(t["computed_tokens"]) / 1e6
        lines.append(
            f"| {name} | {float(t['hbm_share']):.2%} | "
            f"{float(t['tier_share']):.2%} | {float(t['hit_share']):.2%} | "
            f"{float(t['computed_share']):.2%} ({comp_m:.2f}M) | "
            f"{int(t['tier_served_requests'])} |")
    if lanes is not None:
        lines.extend([
            "",
            f"### Little's Law Lane Breakdown (`{lanes.regime}`)",
            "",
            (f"- **Client concurrency**: `{lanes.client_concurrency}` lanes, "
             f"**decode cap**: `{lanes.decode_batch_cap}`, "
             f"**throughput**: `{lanes.throughput_req_s:.3f} req/s`"),
            (f"- **Lanes**: prefill `{lanes.prefill_lanes:.2f}`, "
             f"admission wait `{lanes.admission_wait_lanes:.2f}`, "
             f"KV pull `{lanes.kv_pull_lanes:.2f}`, "
             f"decode generation `{lanes.decode_gen_lanes:.2f}` "
             f"(`{lanes.decode_fill_ratio:.0%}` of cap), "
             f"client think/idle `{lanes.client_think_gap_lanes:.2f}`"),
        ])
    return "\n".join(lines) + "\n"
