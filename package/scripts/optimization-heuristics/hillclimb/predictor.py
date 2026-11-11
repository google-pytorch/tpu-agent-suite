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
"""Closed queueing-network throughput and interactivity predictor (`predictor.py`).

Provides an exact 2-station closed product-form convolution solver (`solve_network`)
for disaggregated P/D topologies (`1P1D`, `1P2D`, `2P1D`, `1P3D`) and a
shared-chip chunked-prefill fixed-point solver for single-host (`1H`) serving.
Complements `workload-microbenchmark`'s fixed-step `T_mix` rollup for
schedule-changing levers (`DECODE_BATCH_CAP`, speculative token count, chunked
prefill budget, admission, and P/D leg balance) where step counts and queue
residencies shift end-to-end.
In-sample fit error of the default parameters (the `WorkloadSlice` tables below
are per-concurrency means of the same runs; there is no held-out split):
  * `1.35%` `tok/s/chip` MAE and `1.08%` `p90` interactivity MAE across 24
    `GLM-5.2-FP8` disaggregated P/D runs (`c=32..96`).
  * `1.95%` `tok/s/chip` MAE across 17 `Qwen3.5-397B-A17B-FP8` single-host runs,
    including closed-loop Little's-law elasticity and KV-cache thrash collapse.
"""

import ast
import dataclasses
import json
import math
import pathlib
import re
import statistics
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from hillclimb import metrics as metrics_lib
from hillclimb import model_plugins

POWER_OF_TWO_BUCKETS = tuple(16 << i for i in range(8))
DEFAULT_EXTRA_BUCKETS = (4, 8)
EP_BUCKETS_FROM = 80
MOE_AUTO_EP_AT = 24
CTX_REF_TOKENS = 150_000.0

OLD_TP = "old_tp"
B2V3 = "b2v3"
B2V3_TRACE = "b2v3_trace"
EP = "ep"
SINGLE_HOST_DEFAULT = "1h_default"


def decode_buckets(extra: Sequence[int] | None, moe: str,
                   step_tokens: int) -> tuple[int, ...]:
    """Returns sorted compiled decode token buckets."""
    buckets = set(POWER_OF_TWO_BUCKETS)
    if extra is not None:
        buckets |= set(extra)
    else:
        buckets |= set(DEFAULT_EXTRA_BUCKETS)
        if moe == "ep":
            buckets |= set(range(EP_BUCKETS_FROM, step_tokens + 16, 16))
    return tuple(sorted(buckets))


def bucket_rows(tokens: int, buckets: Sequence[int]) -> int:
    """Returns smallest compiled bucket `>= tokens` (or `tokens` if above max)."""
    for b in sorted(buckets):
        if b >= tokens:
            return b
    return tokens


def parse_topology(topology_spec: str,
                   chips_per_slice: int = 8) -> tuple[str, int, int, int]:
    """Parses `'1P1D'`, `'1P2D'`, `'2P1D'`, or `'1H'` -> `(mode, P, D, chips)`."""
    cleaned = topology_spec.strip().upper()
    if cleaned in ("1H", "SINGLE", "SINGLE_HOST"):
        chips = 4 if chips_per_slice == 8 else chips_per_slice
        return "single_host", 0, 1, chips
    m = re.fullmatch(r"(\d+)P(\d+)D", cleaned)
    if not m:
        raise ValueError(
            f"Invalid topology {topology_spec!r}; expected e.g. 1P1D, 1P2D, "
            "2P1D, or 1H.")
    n_p = int(m.group(1))
    n_d = int(m.group(2))
    if n_p < 1 or n_d < 1:
        raise ValueError(f"P and D counts must be >= 1 in {topology_spec!r}.")
    return "disagg_pd", n_p, n_d, (n_p + n_d) * chips_per_slice


def _literal(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def _parse_semicolon_env(value: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in str(value or "").split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            if k.strip():
                out[k.strip()] = v.strip()
    return out


def _parse_ints(value: str | None) -> tuple[int, ...]:
    out: list[int] = []
    for tok in str(value or "").replace(";", ",").split(","):
        tok = tok.strip()
        if tok.isdigit():
            out.append(int(tok))
    return tuple(out)


@dataclasses.dataclass(frozen=True)
class CalibratedRun:
    """Normalized completed benchmark run for P/D or single-host calibration."""

    run_id: str
    mode: str
    concurrency: int
    cap: int
    chips: int
    num_prefills: int
    num_decodes: int
    family: str
    buckets: tuple[int, ...]
    kv_prefetch: bool
    proxy_inflight: int
    dcp: int
    al: float
    synthetic_al: bool
    duration_s: float
    kv_cache_hit_rate: float
    num_knobs: int
    isl: float
    osl: float
    new_tokens: float
    req_s: float
    tok_s_chip: float
    itl_mean_ms: float
    itl_median_ms: float
    itl_p90_ms: float
    p90_interactivity: float
    ttft_mean_ms: float
    e2e_mean_ms: float

    @property
    def idle_s(self) -> float:
        return self.concurrency / self.req_s - self.e2e_mean_ms / 1e3


def classify_pd_family(knobs: Mapping[str, str], cap: int,
                       per_decode_conc: int) -> str:
    """Classifies a P/D run's decode step family (`old_tp`, `b2v3`, or `ep`)."""
    moe = knobs.get("DECODE_MOE", "").lower()
    if not moe:
        moe = ("ep" if min(per_decode_conc, max(cap, 16)) >= MOE_AUTO_EP_AT
               else "tp")
    if moe == "ep":
        return EP
    dec_env = _parse_semicolon_env(knobs.get("EXTRA_ENV_DECODE"))
    if dec_env.get("TPU_FUSED_NORM_QKVA") == "1":
        return B2V3
    return OLD_TP


def run_from_jsonl_row(row: Mapping[str, Any]) -> CalibratedRun | None:
    """Parses one succeeded benchmark history JSONL row."""
    status = str(row.get("status") or row.get("job_status") or "Succeeded")
    if status != "Succeeded":
        return None
    m = _literal(row.get("metrics")) or {}
    k = _literal(row.get("knobs")) or {}
    if not isinstance(m, Mapping) or not m.get("token_throughput_per_chip"):
        return None
    if not m.get("request_throughput"):
        return None
    knobs = ({
        str(a): str(b)
        for a, b in k.items()
    } if isinstance(k, Mapping) else {})
    if knobs.get("EXIT_ON_READY", "").lower() == "true":
        return None

    chips = int(m.get("num_chips") or 16)
    conc = int(m.get("max_concurrency") or m.get("conc") or 16)
    recipe = str(row.get("recipe") or "")
    is_single = (chips <= 4 or "qwen3" in recipe
                 or knobs.get("NUM_PREFILLS") == "0")

    if is_single:
        mode = "single_host"
        num_prefills = 0
        num_decodes = 1
        cap = int(
            knobs.get("MAX_NUM_SEQS") or knobs.get("DECODE_BATCH_CAP") or conc)
        family = SINGLE_HOST_DEFAULT
        buckets = decode_buckets(None, "ep", cap * 4)
        kv_prefetch = False
        proxy_inflight = cap
        dcp = 1
        default_al = 3.39
    else:
        mode = "disagg_pd"
        num_decodes = int(knobs.get("NUM_DECODES", "1"))
        num_prefills = int(knobs.get("NUM_PREFILLS", "1"))
        cap = int(knobs.get("DECODE_BATCH_CAP", "48"))
        family = classify_pd_family(knobs, cap, conc // max(1, num_decodes))
        moe = "ep" if family == EP else "tp"
        extra = (_parse_ints(knobs["DECODE_TOKEN_BUCKET_EXTRA"])
                 if "DECODE_TOKEN_BUCKET_EXTRA" in knobs else None)
        buckets = decode_buckets(extra, moe, cap * 4)
        dec_env = _parse_semicolon_env(knobs.get("EXTRA_ENV_DECODE"))
        kv_prefetch = (
            "kv_prefetch_scheduler" in knobs.get("EXTRA_ARGS_DECODE", "")
            and int(dec_env.get("TPU_SCHED_KV_PREFETCH_WAITING", "0")) > 0)
        proxy_inflight = int(knobs.get("PROXY_DECODE_MAX_INFLIGHT", str(cap)))
        dcp = int(knobs.get("DECODE_DCP_SIZE") or knobs.get("DCP_SIZE") or 16)
        default_al = 2.984

    synthetic = knobs.get("SYNTHETIC_ACCEPTANCE_LENGTH", "2.99") != "0"
    isl = float(m["mean_input_tokens"])
    hit = float(m.get("kv_cache_hit_rate") or 0.0)
    itl_med = float(m.get("median_itl_ms") or m.get("median_tpot_ms") or 0.0)
    itl_mean = float(m.get("mean_itl_ms") or m.get("mean_tpot_ms") or itl_med)
    itl_p90 = float(m.get("p90_itl_ms") or m.get("p90_tpot_ms") or itl_mean)
    return CalibratedRun(
        run_id=str(row.get("run_id") or row.get("id") or ""),
        mode=mode,
        concurrency=conc,
        cap=cap,
        chips=chips,
        num_prefills=num_prefills,
        num_decodes=num_decodes,
        family=family,
        buckets=buckets,
        kv_prefetch=kv_prefetch,
        proxy_inflight=proxy_inflight,
        dcp=dcp,
        al=float(m.get("spec_decode_mean_acceptance_length") or default_al),
        synthetic_al=synthetic,
        duration_s=float(m.get("benchmark_duration_s") or 3600.0),
        kv_cache_hit_rate=hit,
        num_knobs=len(knobs),
        isl=isl,
        osl=float(m["mean_output_tokens"]),
        new_tokens=isl * (1.0 - hit),
        req_s=float(m["request_throughput"]),
        tok_s_chip=float(m["token_throughput_per_chip"]),
        itl_mean_ms=itl_mean,
        itl_median_ms=itl_med,
        itl_p90_ms=itl_p90,
        p90_interactivity=float(m["p90_interactivity_tok_s"]),
        ttft_mean_ms=float(m["mean_ttft_ms"]),
        e2e_mean_ms=float(m["mean_e2el_ms"]),
    )


def load_runs_jsonl(path: str | pathlib.Path) -> list[CalibratedRun]:
    """Loads all succeeded benchmark runs with metrics from a JSONL file."""
    runs: list[CalibratedRun] = []
    with pathlib.Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rec = run_from_jsonl_row(json.loads(line))
                if rec is not None:
                    runs.append(rec)
    return runs


@dataclasses.dataclass(frozen=True)
class StepParams:
    """Decode step latency model `s(m) = t0 + a*m*ctx + b*rows + moe` (ms)."""

    t0_ms: float
    a_ms: float
    b_ms: float
    moe_ms: float = 0.0
    moe_cov_ms: float = 0.0
    moe_ref_rows: int = 64
    row_ctx_share: float = 0.0
    above64_ms: float = 0.0


@dataclasses.dataclass(frozen=True)
class WorkloadSlice:
    """Per-concurrency multi-turn workload slice and closed-loop think `Z`."""

    isl: float
    osl: float
    new_tokens: float
    idle_s: float
    e2e_ref_s: float = 0.0
    n_runs: int = 0


@dataclasses.dataclass(frozen=True)
class PDParams:
    """Calibrated parameters for disaggregated P/D (`xPyD`) prediction."""

    step: Mapping[str, StepParams]
    prefill_us_per_wtoken: float
    prefill_ctx_l0: float
    fixed_s: float
    kv_gap_s: float
    p90_jitter: float
    p90_spread: float
    ctx_share: float
    idle_slope: float
    workloads: Mapping[int, WorkloadSlice]
    prefetch_gap_share: float = 0.0
    decode_wait_factor: float = 1.0


@dataclasses.dataclass(frozen=True)
class SingleHostParams:
    """Calibrated parameters for single-host mixed (`1H`) serving."""

    step: StepParams
    prefill_us_per_new_token: float
    base_ttft_ms: float
    chunk_ref_tokens: int
    chunk_tail_ms_at_ref: float
    decode_priority_tail_scale: float
    p90_jitter: float
    idle_slope: float
    kv_token_budget: float
    workloads: Mapping[int, WorkloadSlice]


@dataclasses.dataclass(frozen=True)
class PredictConfig:
    """Configuration evaluated by `predict_serving`."""

    plugin_name: str = model_plugins.DEFAULT_PLUGIN_NAME
    topology: str = "1P1D"
    concurrency: int = 48
    cap: int = 16
    family: str = ""
    buckets: tuple[int, ...] | None = None
    mtp_k: int | None = None
    al: float | None = None
    chips: int | None = None
    kv_prefetch: bool = True
    proxy_inflight: int = 0
    step_delta_ms: float = 0.0
    row_adjust_ms: Mapping[int,
                           float] = dataclasses.field(default_factory=dict)
    prefill_scale: float = 1.0
    chunk_tokens: int = 4096
    decode_priority: bool = True
    workload_override: WorkloadSlice | None = None
    # The caller's chip-hour rate in USD. Only `mtok_per_usd` uses it, and this
    # package ships no default price.
    chip_hour_usd: float | None = None


def _mtok_per_usd(tok_chip: float, cfg: PredictConfig) -> float | None:
    """`Mtok/$` at the caller's chip-hour rate, or `None` without one."""
    if cfg.chip_hour_usd is None:
        return None
    return metrics_lib.mtok_per_usd(tok_chip, cfg.chip_hour_usd)


@dataclasses.dataclass(frozen=True)
class PredictionResult:
    """Predicted serving KPIs and bottleneck leg diagnosis."""

    plugin_name: str
    topology: str
    concurrency: int
    cap: int
    chips: int
    req_s: float
    tok_s_chip: float
    mtok_per_usd: float | None
    itl_median_ms: float
    itl_mean_ms: float
    itl_p90_ms: float
    p90_interactivity: float
    ttft_ms: float
    e2e_ms: float
    step_at_cap_ms: float
    prefill_util: float
    decode_at_cap: float
    idle_s: float
    e2e_elasticity: float
    binding_leg: str
    kv_thrash_risk: bool = False

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def render_markdown(self) -> str:
        thrash = " (**KV THRASH RISK**)" if self.kv_thrash_risk else ""
        if self.mtok_per_usd is None:
            throughput = (f"- **Throughput**: **{self.tok_s_chip:,.1f} "
                          f"tok/s/chip** (`{self.req_s:.3f} req/s`; pass "
                          "`--chip-hour-usd` for Mtok/$)")
        else:
            throughput = (f"- **Throughput**: **{self.mtok_per_usd:.2f} "
                          f"Mtok/$** (`{self.tok_s_chip:,.1f} tok/s/chip`, "
                          f"`{self.req_s:.3f} req/s`)")
        return ("\n".join([
            (f"### Prediction: `{self.plugin_name}` `{self.topology}` "
             f"`C={self.concurrency}` `cap={self.cap}` ({self.chips} chips)"
             f"{thrash}"),
            "",
            throughput,
            (f"- **Interactivity**: **p90 = {self.p90_interactivity:.1f} "
             f"tok/s** (`med ITL {self.itl_median_ms:.2f} ms`, `mean ITL "
             f"{self.itl_mean_ms:.2f} ms`, `p90 ITL {self.itl_p90_ms:.2f} ms`)"
             ),
            (f"- **Latency & Queueing**: `TTFT {self.ttft_ms:.0f} ms`, "
             f"`E2E {self.e2e_ms / 1e3:.2f} s`, `idle Z {self.idle_s:.2f} s`, "
             f"`step@cap {self.step_at_cap_ms:.2f} ms`"),
            ("- **Utilization & Leg**: "
             f"`prefill_util={self.prefill_util:.2f}`, "
             f"`decode_at_cap={self.decode_at_cap:.2f}`, "
             f"`e2e_elasticity={self.e2e_elasticity:.2f}`, "
             f"**binding_leg = `{self.binding_leg}`**"),
        ]) + "\n")


def _logsumexp(xs: Iterable[float]) -> float:
    vals = [x for x in xs if x != -math.inf]
    if not vals:
        return -math.inf
    top = max(vals)
    return top + math.log(sum(math.exp(x - top) for x in vals))


def _log_conv(a: Sequence[float], b: Sequence[float], n: int) -> list[float]:
    return [
        _logsumexp([a[j] + b[m - j] for j in range(m + 1)])
        for m in range(n + 1)
    ]


@dataclasses.dataclass(frozen=True)
class NetworkSolution:
    """Stationary solution of the closed delay + prefill + decode network."""

    throughput: float
    mean_prefill: float
    mean_decode: float
    decode_dist: tuple[float, ...]


def _solve_network_log(
    n: int,
    delay_s: float,
    prefill_rates: Sequence[float],
    decode_rates: Sequence[float],
) -> NetworkSolution:
    log_fd = [0.0]
    for r in decode_rates[:n]:
        log_fd.append(log_fd[-1] - math.log(r))
    log_fp = [0.0]
    for r in prefill_rates[:n]:
        log_fp.append(log_fp[-1] - math.log(r))
    log_d = math.log(delay_s) if delay_s > 0 else -math.inf
    log_fz = [(k * log_d if k else 0.0) - math.lgamma(k + 1)
              for k in range(n + 1)]
    h_pz = _log_conv(log_fp, log_fz, n)
    h_dz = _log_conv(log_fd, log_fz, n)
    log_g = _logsumexp(log_fd[k] + h_pz[n - k] for k in range(n + 1))
    log_g1 = _logsumexp(log_fd[k] + h_pz[n - 1 - k] for k in range(n))
    x = math.exp(log_g1 - log_g)
    pd = [math.exp(log_fd[k] + h_pz[n - k] - log_g) for k in range(n + 1)]
    pp = [math.exp(log_fp[j] + h_dz[n - j] - log_g) for j in range(n + 1)]
    return NetworkSolution(
        throughput=x,
        mean_prefill=sum(j * p for j, p in enumerate(pp)),
        mean_decode=sum(k * p for k, p in enumerate(pd)),
        decode_dist=tuple(pd),
    )


def solve_network(
    customers: int,
    delay_s: float,
    prefill_rates: Sequence[float],
    decode_rates: Sequence[float],
) -> NetworkSolution:
    """Solves the closed product-form queueing network by exact convolution."""
    n = customers
    ref = max(decode_rates[min(n, len(decode_rates)) - 1], 1e-9)
    fd = [1.0]
    for r in decode_rates[:n]:
        fd.append(fd[-1] * ref / r)
    fp = [1.0]
    for r in prefill_rates[:n]:
        fp.append(fp[-1] * ref / r)
    fz = [1.0]
    for k in range(1, n + 1):
        fz.append(fz[-1] * delay_s * ref / k)
    if not all(math.isfinite(t) and t < 1e250 for t in fd + fp + fz):
        return _solve_network_log(n, delay_s, prefill_rates, decode_rates)

    h_pz = [sum(fp[j] * fz[m - j] for j in range(m + 1)) for m in range(n + 1)]
    h_dz = [sum(fd[j] * fz[m - j] for j in range(m + 1)) for m in range(n + 1)]
    g = sum(fd[k] * h_pz[n - k] for k in range(n + 1))
    g1 = sum(fd[k] * h_pz[n - 1 - k] for k in range(n))
    pd = [fd[k] * h_pz[n - k] / g for k in range(n + 1)]
    pp = [fp[j] * h_dz[n - j] / g for j in range(n + 1)]
    return NetworkSolution(
        throughput=ref * g1 / g,
        mean_prefill=sum(j * p for j, p in enumerate(pp)),
        mean_decode=sum(k * p for k, p in enumerate(pd)),
        decode_dist=tuple(pd),
    )


def interpolate_workload(table: Mapping[int, WorkloadSlice],
                         concurrency: int) -> WorkloadSlice:
    """Interpolates `WorkloadSlice` linearly across calibrated concurrencies."""
    if concurrency in table:
        return table[concurrency]
    levels = sorted(table)
    if not levels:
        raise ValueError("Empty workload table in predictor params.")
    if concurrency <= levels[0]:
        return table[levels[0]]
    if concurrency >= levels[-1]:
        return table[levels[-1]]
    lo = max(c for c in levels if c < concurrency)
    hi = min(c for c in levels if c > concurrency)
    w = (concurrency - lo) / (hi - lo)
    a, b = table[lo], table[hi]
    return WorkloadSlice(
        isl=(1.0 - w) * a.isl + w * b.isl,
        osl=(1.0 - w) * a.osl + w * b.osl,
        new_tokens=(1.0 - w) * a.new_tokens + w * b.new_tokens,
        idle_s=(1.0 - w) * a.idle_s + w * b.idle_s,
        e2e_ref_s=(1.0 - w) * a.e2e_ref_s + w * b.e2e_ref_s,
        n_runs=0,
    )


def evaluate_step_ms(
    m: float,
    *,
    mtp_k: int,
    buckets: Sequence[int],
    sp: StepParams,
    step_delta_ms: float = 0.0,
    row_adjust_ms: Mapping[int, float] | None = None,
    ctx_factor: float = 1.0,
    ctx_ratio: float = 1.0,
    num_experts: int = 256,
    top_k: int = 8,
) -> float:
    """Evaluates pure decode step time (ms) at `m` running requests."""
    if m <= 0:
        return 0.0
    lo = math.floor(m)
    hi = math.ceil(m)
    if lo != hi and lo >= 1:
        w = m - lo
        s_lo = evaluate_step_ms(
            lo,
            mtp_k=mtp_k,
            buckets=buckets,
            sp=sp,
            step_delta_ms=step_delta_ms,
            row_adjust_ms=row_adjust_ms,
            ctx_factor=ctx_factor,
            ctx_ratio=ctx_ratio,
            num_experts=num_experts,
            top_k=top_k,
        )
        s_hi = evaluate_step_ms(
            hi,
            mtp_k=mtp_k,
            buckets=buckets,
            sp=sp,
            step_delta_ms=step_delta_ms,
            row_adjust_ms=row_adjust_ms,
            ctx_factor=ctx_factor,
            ctx_ratio=ctx_ratio,
            num_experts=num_experts,
            top_k=top_k,
        )
        return (1.0 - w) * s_lo + w * s_hi

    n = max(1, int(round(m)))
    live = (1 + mtp_k) * n
    rows = bucket_rows(live, buckets)
    row_factor = (1.0 - sp.row_ctx_share) + sp.row_ctx_share * ctx_ratio
    moe = 0.0
    if sp.moe_ms:
        cov_live = model_plugins.expected_active_experts(
            live, num_experts, top_k) / max(1, num_experts)
        cov_ref = model_plugins.expected_active_experts(
            sp.moe_ref_rows, num_experts, top_k) / max(1, num_experts)
        moe = sp.moe_ms + sp.moe_cov_ms * (cov_live - cov_ref)

    adj = row_adjust_ms.get(rows, 0.0) if row_adjust_ms else 0.0
    raw = (sp.t0_ms - step_delta_ms + sp.a_ms * n * ctx_factor +
           sp.b_ms * rows * row_factor + moe +
           (sp.above64_ms if rows > 64 else 0.0) + adj)
    return max(1.0, raw)


GLM5_DEFAULT_PD_PARAMS = PDParams(
    step={
        OLD_TP:
        StepParams(t0_ms=10.9438, a_ms=0.3983, b_ms=0.2253),
        B2V3:
        StepParams(t0_ms=10.9438 - 3.7112, a_ms=0.3983, b_ms=0.2253),
        B2V3_TRACE:
        StepParams(
            t0_ms=10.02,
            a_ms=0.10,
            b_ms=0.083,
            moe_ms=11.37,
            moe_cov_ms=8.01,
            moe_ref_rows=64,
            row_ctx_share=0.30,
            above64_ms=0.35,
        ),
        EP:
        StepParams(t0_ms=4.7155, a_ms=1.3537, b_ms=0.0300),
    },
    prefill_us_per_wtoken=23.6761,
    prefill_ctx_l0=62_500.0,
    fixed_s=1.9506,
    kv_gap_s=0.4350,
    p90_jitter=1.0000,
    p90_spread=0.5350,
    ctx_share=0.20,
    idle_slope=0.3673,
    workloads={
        32:
        WorkloadSlice(
            isl=156_666.8,
            osl=1004.3,
            new_tokens=6243.2,
            idle_s=16.2108,
            e2e_ref_s=15.2926,
            n_runs=2,
        ),
        48:
        WorkloadSlice(
            isl=148_361.9,
            osl=1027.8,
            new_tokens=6515.7,
            idle_s=15.9908,
            e2e_ref_s=17.8741,
            n_runs=9,
        ),
        56:
        WorkloadSlice(
            isl=139_608.2,
            osl=1044.3,
            new_tokens=7041.8,
            idle_s=11.2188,
            e2e_ref_s=28.5977,
            n_runs=1,
        ),
        64:
        WorkloadSlice(
            isl=127_066.6,
            osl=995.2,
            new_tokens=6999.3,
            idle_s=10.0535,
            e2e_ref_s=24.9085,
            n_runs=11,
        ),
        96:
        WorkloadSlice(
            isl=97_288.1,
            osl=997.3,
            new_tokens=6639.9,
            idle_s=16.9979,
            e2e_ref_s=22.2109,
            n_runs=1,
        ),
    },
)

# No fit error is reported for these `deepseek_mla_moe` defaults (the MAE in the
# module docstring covers GLM-5.2 and Qwen3.5 only); calibrate them with
# `--history` before relying on them.
DEEPSEEK_DEFAULT_PD_PARAMS = dataclasses.replace(
    GLM5_DEFAULT_PD_PARAMS,
    step={
        B2V3: StepParams(t0_ms=6.80, a_ms=0.36, b_ms=0.19),
        EP: StepParams(t0_ms=4.40, a_ms=1.18, b_ms=0.03),
        OLD_TP: StepParams(t0_ms=9.90, a_ms=0.38, b_ms=0.20),
    },
    prefill_us_per_wtoken=21.5,
)

QWEN35_DEFAULT_1H_PARAMS = SingleHostParams(
    step=StepParams(t0_ms=10.95, a_ms=0.185, b_ms=0.059),
    prefill_us_per_new_token=32.6,
    base_ttft_ms=185.0,
    chunk_ref_tokens=4096,
    chunk_tail_ms_at_ref=18.0,
    decode_priority_tail_scale=0.82,
    p90_jitter=1.08,
    idle_slope=0.12,
    kv_token_budget=2_400_000.0,
    workloads={
        8:
        WorkloadSlice(
            isl=99_147.2,
            osl=996.5,
            new_tokens=5790.2,
            idle_s=15.669,
            e2e_ref_s=3.177,
            n_runs=1,
        ),
        16:
        WorkloadSlice(
            isl=107_716.1,
            osl=937.1,
            new_tokens=6505.1,
            idle_s=14.333,
            e2e_ref_s=4.218,
            n_runs=16,
        ),
    },
)


def default_pd_params(plugin_name: str) -> PDParams:
    plugin = model_plugins.get_plugin(plugin_name)
    if plugin.name == "deepseek_mla_moe":
        return DEEPSEEK_DEFAULT_PD_PARAMS
    return GLM5_DEFAULT_PD_PARAMS


def _diagnose_pd_leg(
    *,
    prefill_util: float,
    decode_at_cap: float,
    decode_wait_s: float,
    idle_s: float,
    e2e_s: float,
    kv_prefetch: bool,
    gap_s: float,
) -> str:
    if not kv_prefetch and gap_s >= 0.25:
        return "kv_handoff_bound"
    if prefill_util >= 0.85 and decode_at_cap < 0.35:
        return "prefill_bound"
    if decode_at_cap >= 0.45 or decode_wait_s >= 1.5:
        return "decode_cap_bound"
    if idle_s / max(1e-6, idle_s + e2e_s) >= 0.58 and prefill_util < 0.75:
        return "client_idle_bound"
    return "balanced_pd"


def predict_disagg_pd(
    cfg: PredictConfig,
    params: PDParams | None = None,
    max_iter: int = 60,
) -> PredictionResult:
    """Predicts serving KPIs for a disaggregated P/D (`xPyD`) configuration."""
    plugin = model_plugins.get_plugin(cfg.plugin_name)
    pd_params = params or default_pd_params(plugin.name)
    _, num_p, num_d, default_chips = parse_topology(
        cfg.topology, plugin.topology.chips_per_slice)
    chips = cfg.chips or default_chips
    mtp_k = cfg.mtp_k if cfg.mtp_k is not None else plugin.dims.mtp_draft_tokens
    al = cfg.al if cfg.al is not None else plugin.default_acceptance_length

    per_dec_conc = cfg.concurrency // max(1, num_d)
    if cfg.family:
        family = cfg.family
    else:
        family = (EP if min(per_dec_conc, max(cfg.cap, 16)) >= MOE_AUTO_EP_AT
                  else B2V3)
    if family not in pd_params.step:
        family = B2V3 if B2V3 in pd_params.step else next(iter(pd_params.step))
    sp = pd_params.step[family]
    moe = "ep" if family == EP else "tp"
    buckets = cfg.buckets or decode_buckets(None, moe, cfg.cap * (1 + mtp_k))

    wl = cfg.workload_override or interpolate_workload(pd_params.workloads,
                                                       cfg.concurrency)
    ctx_ratio = wl.isl / CTX_REF_TOKENS
    ctx_factor = (1.0 - pd_params.ctx_share) + pd_params.ctx_share * ctx_ratio
    steps_per_req = max(wl.osl - 1.0, 1.0) / al
    slots = cfg.cap * num_d
    proxy = cfg.proxy_inflight or (cfg.cap + 2)
    hidden = cfg.kv_prefetch and proxy > cfg.cap
    gap = pd_params.kv_gap_s * (pd_params.prefetch_gap_share
                                if hidden else 1.0)
    n = cfg.concurrency

    def step_at(m_total: float) -> float:
        return evaluate_step_ms(
            m_total / num_d,
            mtp_k=mtp_k,
            buckets=buckets,
            sp=sp,
            step_delta_ms=cfg.step_delta_ms,
            row_adjust_ms=cfg.row_adjust_ms,
            ctx_factor=ctx_factor,
            ctx_ratio=ctx_ratio,
            num_experts=plugin.dims.num_experts or 256,
            top_k=plugin.dims.top_k or 8,
        )

    decode_rates = []
    for k in range(1, n + 1):
        m = min(k, slots)
        hold = steps_per_req * step_at(m) / 1e3 + gap
        decode_rates.append(m / hold)

    s_p = (cfg.prefill_scale * pd_params.prefill_us_per_wtoken * 1e-6 *
           wl.new_tokens * (1.0 + wl.isl / pd_params.prefill_ctx_l0))
    prefill_rates = [min(k, num_p) / s_p for k in range(1, n + 1)]

    zeta = pd_params.idle_slope if wl.e2e_ref_s else 0.0
    v = pd_params.decode_wait_factor
    idle, extra = wl.idle_s, 0.0
    sol = solve_network(n, idle + pd_params.fixed_s, prefill_rates,
                        decode_rates)
    for _ in range(max_iter if (zeta or v != 1.0) else 0):
        x_ = sol.throughput
        mean_run_ = sum(
            min(k, slots) * p for k, p in enumerate(sol.decode_dist))
        w_dec = (sol.mean_decode - mean_run_) / x_
        e2e = n / x_ - idle - extra + (v - 1.0) * w_dec
        new_idle = max(0.5, wl.idle_s - zeta * (e2e - wl.e2e_ref_s))
        new_extra = max(0.0, (v - 1.0) * w_dec)
        if abs(new_idle - idle) < 1e-4 and abs(new_extra - extra) < 1e-4:
            break
        idle, extra = new_idle, new_extra
        sol = solve_network(n, idle + extra + pd_params.fixed_s, prefill_rates,
                            decode_rates)

    x = sol.throughput
    pd_dist = sol.decode_dist
    p_run = [0.0] * (slots + 1)
    for k, p in enumerate(pd_dist):
        p_run[min(k, slots)] += p
    mean_run = sum(m * p for m, p in enumerate(p_run))
    q = [m * p / mean_run for m, p in enumerate(p_run)] if mean_run > 0 else []
    inv = sum(qm / step_at(m) for m, qm in enumerate(q) if m > 0 and qm > 0)
    s_mean = 1.0 / inv if inv > 0 else step_at(1)
    cum, m90 = 0.0, slots
    for m, qm in enumerate(q):
        cum += qm
        if cum >= 0.90:
            m90 = m
            break
    s_hi = step_at(max(m90, 1))
    s_p90 = pd_params.p90_jitter * (s_mean + pd_params.p90_spread *
                                    (s_hi - s_mean))
    itl_med = s_mean / al
    itl_p90 = s_p90 / al
    t_serv = mean_run / x
    wait_dec = (sol.mean_decode - mean_run) / x + extra
    r_prefill = sol.mean_prefill / x
    ttft_s = r_prefill + pd_params.fixed_s + wait_dec + gap
    e2e_s = ttft_s + t_serv - gap
    tok_chip = x * (wl.isl + wl.osl) / chips
    prefill_util = x * s_p / num_p
    step_cap = step_at(slots)

    e2e_share = e2e_s / max(1e-6, idle + e2e_s)
    dec_share = max(0.0, t_serv - gap) / max(1e-6, e2e_s)
    elasticity = -e2e_share * dec_share * (1.0 - min(0.85, prefill_util**3))

    leg = _diagnose_pd_leg(
        prefill_util=prefill_util,
        decode_at_cap=p_run[slots],
        decode_wait_s=wait_dec,
        idle_s=idle,
        e2e_s=e2e_s,
        kv_prefetch=hidden,
        gap_s=gap,
    )
    return PredictionResult(
        plugin_name=plugin.name,
        topology=f"{num_p}P{num_d}D",
        concurrency=cfg.concurrency,
        cap=cfg.cap,
        chips=chips,
        req_s=x,
        tok_s_chip=tok_chip,
        mtok_per_usd=_mtok_per_usd(tok_chip, cfg),
        itl_median_ms=itl_med,
        itl_mean_ms=itl_med,
        itl_p90_ms=itl_p90,
        p90_interactivity=1e3 / itl_p90,
        ttft_ms=ttft_s * 1e3,
        e2e_ms=e2e_s * 1e3,
        step_at_cap_ms=step_cap,
        prefill_util=prefill_util,
        decode_at_cap=p_run[slots],
        idle_s=idle,
        e2e_elasticity=elasticity,
        binding_leg=leg,
        kv_thrash_risk=False,
    )


def predict_single_host(
    cfg: PredictConfig,
    params: SingleHostParams | None = None,
    max_iter: int = 50,
) -> PredictionResult:
    """Predicts KPIs for single-host mixed (`1H`) serving with chunked prefill."""
    plugin = model_plugins.get_plugin(cfg.plugin_name)
    sh_params = params or QWEN35_DEFAULT_1H_PARAMS
    chips = cfg.chips or plugin.topology.chips_per_slice
    mtp_k = cfg.mtp_k if cfg.mtp_k is not None else plugin.dims.mtp_draft_tokens
    al = cfg.al if cfg.al is not None else plugin.default_acceptance_length
    buckets = cfg.buckets or decode_buckets(None, "ep", cfg.cap * (1 + mtp_k))
    wl = cfg.workload_override or interpolate_workload(sh_params.workloads,
                                                       cfg.concurrency)

    active_cap = min(cfg.concurrency, cfg.cap)
    kv_demand_tokens = active_cap * wl.isl
    kv_thrash = kv_demand_tokens > sh_params.kv_token_budget
    thrash_multiplier = (1.0 + 3.5 *
                         ((kv_demand_tokens / sh_params.kv_token_budget) - 1.0)
                         if kv_thrash else 1.0)

    s_prefill_req = (cfg.prefill_scale * sh_params.prefill_us_per_new_token *
                     1e-6 * wl.new_tokens * thrash_multiplier)

    idle = wl.idle_s
    x = cfg.concurrency / max(1.0, idle + (wl.e2e_ref_s or 4.0))
    m_run = min(float(cfg.cap), max(1.0, x * 3.5))
    rho_p = min(0.88, x * s_prefill_req)
    pure_step_ms = 12.0
    ttft_ms = sh_params.base_ttft_ms
    e2e_s = 4.0

    for _ in range(max_iter):
        pure_step_ms = (evaluate_step_ms(
            m_run,
            mtp_k=mtp_k,
            buckets=buckets,
            sp=sh_params.step,
            step_delta_ms=cfg.step_delta_ms,
            row_adjust_ms=cfg.row_adjust_ms,
            num_experts=plugin.dims.num_experts or 512,
            top_k=plugin.dims.top_k or 10,
        ) * thrash_multiplier)
        itl_med = pure_step_ms / al
        rho_p = min(0.90, max(0.01, x * s_prefill_req))
        itl_mean = itl_med / max(0.10, 1.0 - rho_p)
        wait_factor = 1.0 / max(0.12, 1.0 - 0.65 * rho_p)
        queue_overflow_s = (max(0.0, cfg.concurrency - cfg.cap) *
                            (itl_mean * wl.osl / 1e3) * 0.15)
        ttft_ms = (sh_params.base_ttft_ms + s_prefill_req * 1e3 * wait_factor +
                   queue_overflow_s * 1e3)
        decode_s = max(1.0, wl.osl - 1.0) * itl_mean / 1e3
        e2e_s = ttft_ms / 1e3 + decode_s
        if wl.e2e_ref_s > 0:
            idle = max(
                1.0, wl.idle_s - sh_params.idle_slope * (e2e_s - wl.e2e_ref_s))
        new_x = cfg.concurrency / max(0.5, idle + e2e_s)
        new_m_run = min(float(cfg.cap), max(1.0, new_x * decode_s))
        if abs(new_x - x) < 1e-5 and abs(new_m_run - m_run) < 1e-4:
            x, m_run = new_x, new_m_run
            break
        x = 0.6 * new_x + 0.4 * x
        m_run = 0.6 * new_m_run + 0.4 * m_run

    pure_step_cap_ms = evaluate_step_ms(
        cfg.cap,
        mtp_k=mtp_k,
        buckets=buckets,
        sp=sh_params.step,
        step_delta_ms=cfg.step_delta_ms,
        row_adjust_ms=cfg.row_adjust_ms,
        num_experts=plugin.dims.num_experts or 512,
        top_k=plugin.dims.top_k or 10,
    )
    itl_med = pure_step_ms / al
    itl_mean = itl_med / max(0.10, 1.0 - rho_p)

    chunk_ratio = max(0.25,
                      cfg.chunk_tokens / max(1, sh_params.chunk_ref_tokens))
    prio_scale = (sh_params.decode_priority_tail_scale
                  if cfg.decode_priority else 1.18)
    chunk_tail_ms = (sh_params.chunk_tail_ms_at_ref * (chunk_ratio**0.75) *
                     prio_scale * rho_p * thrash_multiplier)
    itl_p90 = itl_med * sh_params.p90_jitter + chunk_tail_ms
    tok_chip = x * (wl.isl + wl.osl) / chips
    decode_at_cap = min(1.0, max(0.0, (m_run / max(1, cfg.cap))**3))

    decode_s = max(1.0, wl.osl - 1.0) * itl_mean / 1e3
    elasticity = -(decode_s / max(1e-6, idle + e2e_s)) * (1.0 - rho_p)

    if kv_thrash:
        leg = "kv_collapse_bound"
    elif itl_p90 / max(1e-6, itl_med) >= 1.75:
        leg = "chunked_prefill_tail_bound"
    elif idle / max(1e-6, idle + e2e_s) >= 0.65:
        leg = "client_idle_bound"
    else:
        leg = "decode_step_bound"

    return PredictionResult(
        plugin_name=plugin.name,
        topology="1H",
        concurrency=cfg.concurrency,
        cap=cfg.cap,
        chips=chips,
        req_s=x,
        tok_s_chip=tok_chip,
        mtok_per_usd=_mtok_per_usd(tok_chip, cfg),
        itl_median_ms=itl_med,
        itl_mean_ms=itl_mean,
        itl_p90_ms=itl_p90,
        p90_interactivity=1e3 / itl_p90,
        ttft_ms=ttft_ms,
        e2e_ms=e2e_s * 1e3,
        step_at_cap_ms=pure_step_cap_ms,
        prefill_util=rho_p,
        decode_at_cap=decode_at_cap,
        idle_s=idle,
        e2e_elasticity=elasticity,
        binding_leg=leg,
        kv_thrash_risk=kv_thrash,
    )


def predict_serving(
    cfg: PredictConfig,
    *,
    pd_params: PDParams | None = None,
    single_host_params: SingleHostParams | None = None,
) -> PredictionResult:
    """Unified entry point routing `cfg` to `predict_disagg_pd` or `1H`."""
    plugin = model_plugins.get_plugin(cfg.plugin_name)
    topo = cfg.topology.strip()
    if not topo:
        if plugin.topology.mode == "single_host":
            topo = "1H"
        else:
            topo = f"{plugin.topology.prefill_slices}P{plugin.topology.decode_slices}D"
    resolved_cfg = dataclasses.replace(cfg, topology=topo)
    mode, _, _, _ = parse_topology(topo, plugin.topology.chips_per_slice)
    if mode == "single_host":
        return predict_single_host(resolved_cfg, params=single_host_params)
    return predict_disagg_pd(resolved_cfg, params=pd_params)


def measure_workloads_from_runs(
    runs: Sequence[CalibratedRun], ) -> dict[int, WorkloadSlice]:
    """Computes empirical `WorkloadSlice` means per concurrency level."""
    by_c: dict[int, list[CalibratedRun]] = {}
    for r in runs:
        by_c.setdefault(r.concurrency, []).append(r)
    return {
        c:
        WorkloadSlice(
            isl=statistics.fmean(r.isl for r in rs),
            osl=statistics.fmean(r.osl for r in rs),
            new_tokens=statistics.fmean(r.new_tokens for r in rs),
            idle_s=statistics.fmean(r.idle_s for r in rs),
            e2e_ref_s=statistics.fmean(r.e2e_mean_ms / 1e3 for r in rs),
            n_runs=len(rs),
        )
        for c, rs in by_c.items()
    }


def calibrate_from_jsonl(
    jsonl_path: str | pathlib.Path,
    plugin_name: str = model_plugins.DEFAULT_PLUGIN_NAME,
) -> PDParams | SingleHostParams:
    """Calibrates workload slices & step offsets from a run-history JSONL."""
    plugin = model_plugins.get_plugin(plugin_name)
    runs = load_runs_jsonl(jsonl_path)
    if plugin.topology.mode == "single_host":
        core = [
            r for r in runs
            if r.mode == "single_host" and r.kv_cache_hit_rate >= 0.80
        ]
        if not core:
            return QWEN35_DEFAULT_1H_PARAMS
        wls = measure_workloads_from_runs(core)
        base = dataclasses.replace(QWEN35_DEFAULT_1H_PARAMS, workloads=wls)
        diffs: list[float] = []
        for r in core:
            pred = predict_single_host(
                PredictConfig(
                    plugin_name=plugin.name,
                    topology="1H",
                    concurrency=r.concurrency,
                    cap=r.cap,
                    al=r.al,
                ),
                params=base,
            )
            diffs.append((r.itl_median_ms - pred.itl_median_ms) * r.al)
        mean_shift = statistics.fmean(diffs)
        tuned_step = dataclasses.replace(base.step,
                                         t0_ms=max(
                                             1.0,
                                             base.step.t0_ms + mean_shift))
        return dataclasses.replace(base, step=tuned_step)

    core_pd = [
        r for r in runs if r.mode == "disagg_pd" and r.dcp == 16
        and r.synthetic_al and r.duration_s > 3000
    ]
    if not core_pd:
        return default_pd_params(plugin.name)
    wls_pd = measure_workloads_from_runs(core_pd)
    return dataclasses.replace(default_pd_params(plugin.name),
                               workloads=wls_pd)


def parse_microbench_row_deltas_ms(
    log_text: str,
    *,
    calls_per_step: float = 78.0,
) -> dict[int, float]:
    """Parses `RESULT {...}` lines into per-row-bucket step adjustments (ms)."""
    adjustments: dict[int, float] = {}
    for line in log_text.splitlines():
        idx = line.find("RESULT ")
        if idx < 0:
            continue
        payload_str = line[idx + len("RESULT "):].strip()
        try:
            obj = json.loads(payload_str)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, Mapping):
            continue
        rows_val = obj.get("rows") or obj.get("num_tokens") or obj.get("m")
        if rows_val is None:
            continue
        base_us = obj.get("baseline_us") or obj.get("fp8_us") or obj.get(
            "old_us")
        cand_us = obj.get("candidate_us") or obj.get("fp4_us") or obj.get(
            "new_us")
        if base_us is None or cand_us is None:
            continue
        delta_ms = (float(cand_us) - float(base_us)) * calls_per_step / 1e3
        adjustments[int(rows_val)] = (adjustments.get(int(rows_val), 0.0) +
                                      delta_ms)
    return adjustments


def step_delta_from_proxy_tpot(
    baseline_tpot_ms: float,
    candidate_tpot_ms: float,
    acceptance_length: float = 2.984,
) -> float:
    """Converts a single-slice decode proxy TPOT gain into `step_delta_ms`."""
    return (baseline_tpot_ms - candidate_tpot_ms) * acceptance_length


def evaluate_fit_accuracy(
    runs: Sequence[CalibratedRun],
    plugin_name: str,
    *,
    pd_params: PDParams | None = None,
    single_host_params: SingleHostParams | None = None,
) -> dict[str, float]:
    """Computes relative MAE and RMSE on `tok_s_chip` and `p90_interactivity`."""
    err_tpc: list[float] = []
    err_p90: list[float] = []
    for r in runs:
        topo = ("1H" if r.mode == "single_host" else
                f"{r.num_prefills}P{r.num_decodes}D")
        pred = predict_serving(
            PredictConfig(
                plugin_name=plugin_name,
                topology=topo,
                concurrency=r.concurrency,
                cap=r.cap,
                family=r.family,
                buckets=r.buckets,
                al=r.al,
                chips=r.chips,
                kv_prefetch=r.kv_prefetch,
                proxy_inflight=r.proxy_inflight,
            ),
            pd_params=pd_params,
            single_host_params=single_host_params,
        )
        err_tpc.append(pred.tok_s_chip / r.tok_s_chip - 1.0)
        err_p90.append(pred.p90_interactivity / r.p90_interactivity - 1.0)
    if not err_tpc:
        return {"n": 0.0, "tpc_mae": 0.0, "p90_mae": 0.0}
    return {
        "n": float(len(err_tpc)),
        "tpc_mae": statistics.fmean(abs(x) for x in err_tpc),
        "tpc_rmse": math.sqrt(statistics.fmean(x * x for x in err_tpc)),
        "p90_mae": statistics.fmean(abs(x) for x in err_p90),
        "p90_rmse": math.sqrt(statistics.fmean(x * x for x in err_p90)),
    }
